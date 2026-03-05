# 1\_chess\_midterm

This repository is created for Transformers midterm assignment one. Topic "chess".



The implemented method generates a Transformer-based chess player that selects moves using a candidate evaluation algorithm and a language model. First, the current chessboard is represented as a FEN string and passed to a pre-trained Transformer model (Qwen3.5-4B). Using the Python Chess library, the system generates a list of all legal moves and then reduces it to a smaller set of promising candidates (such as promotions, checks, captures, and castling) to improve computational efficiency. For each candidate, the model evaluates the probability of matching the description of the board state by calculating the logarithmic probability of move tokens. The move with the highest average logarithmic probability is selected as the best move. If this evaluation process fails for any reason, the system reverts to the generation procedure, where the model directly generates a legal move. As a final safety mechanism, if the model still fails to generate a legal move, a random legal move is selected to ensure the game continues without error.

