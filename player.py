# -*- coding: utf-8 -*-
"""The implemented method is based on logics of best-move probabilities. As mentioned in comments of code below, we use hierarchy of moves promotions (pawn promotion is crucial in game) > checks > captures > castles > others, to ensure each of them gets a specific score when calculating a logistic probability of the move. Then our player evaluates all the possible moves and scores the highest in the selection. The principle is called a candidate evaluation algorithm in language model. Now in more detail. First, the current chessboard is represented as a FEN string and passed to a pre-trained Transformer model (Qwen3.5-4B). Using the Python Chess library, the system generates a list of all legal moves and then reduces it to a smaller set of promising candidates (such as promotions, checks, captures, and castling) to improve computational efficiency. For each candidate, the model evaluates the probability of matching the description of the board state by calculating the logarithmic probability of move tokens. The move with the highest average logarithmic probability is selected as the best move. If this evaluation process fails for any reason, the system reverts to the generation procedure, where the model directly generates a legal move. As a final safety mechanism, if the model still fails to generate a legal move, a random legal move is selected to ensure the game continues without error."""

from __future__ import annotations #to use Optional[str] hints

import re     #to extract UCI move from model text
import random   #to never crash/return illegal
from typing import Optional, List, Tuple

import chess #to read FEN, generate legal moves etc..
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from chess_tournament import (
    Game,
    Player,
    RandomPlayer,
    LMPlayer,
    SmolPlayer,
    EnginePlayer,
    run_tournament
    )

_UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE) #allowed moves formation


class TransformerPlayer(Player):

   """
   Transformer-based chess player.
    - Input: FEN string
    - Output: UCI move string (e.g., 'e2e4', 'e7e8q') or None if no legal moves

    Upgrade: candidate scoring (reranking) over legal moves for stronger, more stable play.
   """
   def __init__(
        self,
        name: str,
        model_name = "Qwen/Qwen3.5-4B",
        seed: int = 123,  #ensure repeatble results
        use_4bit: bool = True,  #use model in 4-bit to fit in Colab GPU memory
        # generation fallback params
        max_new_tokens: int = 8,
        temperature: float = 0.2,
        top_p: float = 0.9,
        # candidate scoring params for strengthening machine
        use_candidate_scoring: bool = True,     #if true, score a legal move instead of randomly generated
        max_candidates: int = 48,     #score only top N legal moves
        score_batch_size: int = 8,     #batch size for scoring
    ):
        super().__init__(name)

        random.seed(seed)
        self.model_name = model_name

        # Backup generation settings (only used if scoring fails)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Candidate scoring settings
        self.use_candidate_scoring = use_candidate_scoring
        self.max_candidates = max_candidates
        self.score_batch_size = score_batch_size

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True,
        )

        # (4-bit) to avoid syntax mismatch
        quant_cfg = None
        if use_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, #speed
                bnb_4bit_use_double_quant=True,   #extra compression trick
            )

        # Load actual model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quant_cfg,
            trust_remote_code=True,
        )
        self.model.eval()

    # Utilities
   def _legal_moves_uci(self, board: chess.Board) -> List[str]:
        return [m.uci() for m in board.legal_moves]   # convert python-chess Move objects to UCI strings

   def _extract_uci(self, text: str) -> Optional[str]:
        text = text.strip().lower()
        m = _UCI_RE.search(text)
        return m.group(1).lower() if m else None


    # Candidate scoring!
   def _candidate_subset(self, board: chess.Board, legal_uci: List[str]) -> List[str]:
        """
        We don't score ALL legal moves (could be 30-80). It's slow.
        So we keep "more important" moves first:
        promotions > checks > captures > castles > others
        """
        moves = [chess.Move.from_uci(m) for m in legal_uci]

        promos, captures, castles, others = [], [], [], []
        for mv in moves:
            if mv.promotion is not None:
                promos.append(mv) #to ensure pawn promotion is taken into considertion
            elif board.is_capture(mv):
                captures.append(mv)   #critical captures
            elif board.is_castling(mv):
                castles.append(mv)    #castling for king safety
            else:
                others.append(mv)

        checks = []   #this chunk to temporarily play the move and undo it if move gives check
        for mv in moves:
            board.push(mv)
            if board.is_check():
                checks.append(mv)
            board.pop()

        ranked = promos + checks + captures + castles #priority order
        seen = set(ranked)
        ranked += [m for m in others if m not in seen]

        ranked_uci = [m.uci() for m in ranked]
        if len(ranked_uci) > self.max_candidates:
            ranked_uci = ranked_uci[: self.max_candidates]
        return ranked_uci     #list of moves we will actually score

   def _build_scoring_prompt(self, fen: str) -> str:
        # Keep stable and short, this promt is the same for all candidates.
        return (
            "You are a strong chess player.\n"
            "Given this position in FEN:\n"
            f"{fen}\n\n"
            "Best move in UCI:"
        )

   @torch.no_grad()
   def _score_candidates(self, prompt: str, candidates: List[str]) -> List[float]:
        """
        Score each candidate by mean log-prob of its tokens as continuation of the prompt.
        The higher the score, the more model prefers that candidate.
        """
        scores: List[float] = []

        # Prompt length in tokens (candidate starts after this)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.model.device)
        prompt_len = prompt_ids.shape[1]
 # Score candidates in batches (faster on GPU)
        for i in range(0, len(candidates), self.score_batch_size):
            batch = candidates[i : i + self.score_batch_size]
            texts = [f"{prompt} {c}" for c in batch]  # prompt and after a candidate, since leading space helps tokenization

            enc = self.tokenizer(texts, return_tensors="pt", padding=True)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}

            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"] #1 for real tokens, 0 for padding

            out = self.model(input_ids=input_ids, attention_mask=attention_mask)  #forward pass
            logits = out.logits  # raw model scores for next token (B, T, V)
            logprobs = torch.log_softmax(logits, dim=-1)

            for b in range(input_ids.shape[0]):
                ids = input_ids[b]
                mask = attention_mask[b]
                end = int(mask.sum().item())  # non-pad end
                start = prompt_len

                # If something weird happens and we have no candidate tokens
                if start >= end:
                    scores.append(-1e9)
                    continue

                # Candidate tokens are ids[start:end]
                tok = ids[start:end]
                # Pred positions are start-1 ... end-2
                pred_pos = torch.arange(start - 1, end - 1, device=self.model.device)

                lp = logprobs[b, pred_pos, tok]  # (cand_len,)
                scores.append(lp.mean().item())  # length-normalized

        return scores

   def _pick_best_scored_move(self, board: chess.Board, fen: str, legal: List[str]) -> Optional[str]:
        prompt = self._build_scoring_prompt(fen)
        candidates = self._candidate_subset(board, legal)
        if not candidates:
            return None

        cand_scores = self._score_candidates(prompt, candidates)
        best_idx = max(range(len(candidates)), key=lambda k: cand_scores[k])
        best_move = candidates[best_idx]
        return best_move if best_move in legal else None

        # Generation fallback to avoid illegal moves
   def _build_generation_prompt(self, fen: str, legal_moves: List[str]) -> str:
        return (
            "You are an extremely powerful chess engine.\n"
            "Given this chess position in FEN:\n"
            f"{fen}\n\n"
            "Choose exactly ONE move from the following LEGAL moves (UCI):\n"
            f"{', '.join(legal_moves)}\n\n"
            "Return ONLY the move in UCI format. No explanation."
        )

   def _generate_move(self, fen: str, legal: List[str]) -> Optional[str]:
        prompt = self._build_generation_prompt(fen, legal)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=pad_id,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        move = self._extract_uci(text)
        return move if move in legal else None

     # main required method
   def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)    #create board from fen
        legal = self._legal_moves_uci(board)  #list legal moves
        if not legal:
            return None

        # main: candidate scoring
        if self.use_candidate_scoring:
            try:
                best = self._pick_best_scored_move(board, fen, legal)
                if best is not None:
                    return best
            except Exception:
                # if scoring fails, fall back to generation / random
                pass

        # backup: generate + validate
        try:
            move = self._generate_move(fen, legal)
            if move is not None:
                return move
        except Exception:
            pass

        # last resort: random legal move
        return random.choice(legal)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,     #defining new bit config (4bit) for new syntax off model load
    bnb_4bit_use_double_quant=True
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-4B",
    device_map="auto",
    quantization_config=bnb_config
    )

my_player = TransformerPlayer("Student")   # student name, as suggested in the prompt