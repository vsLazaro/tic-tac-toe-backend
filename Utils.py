def minimax(board, depth, is_maximizing):
    winner = check_winner(board)
    if winner == 1:  # Vitória da rede neural
        return 10 - depth
    if winner == -1:  # Vitória do Minimax
        return depth - 10
    if 0 not in board:  # Empate
        return 0

    if is_maximizing:
        max_eval = -float("inf")
        for i in range(9):
            if board[i] == 0:
                board[i] = 1
                eval = minimax(board, depth + 1, False)
                board[i] = 0
                max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float("inf")
        for i in range(9):
            if board[i] == 0:
                board[i] = -1
                eval = minimax(board, depth + 1, True)
                board[i] = 0
                min_eval = min(min_eval, eval)
        return min_eval

def minimax_best_move(board):
    best_score = -float("inf")
    best_move = -1
    for i in range(9):
        if board[i] == 0:
            board[i] = -1
            score = minimax(board, 0, False)
            board[i] = 0
            if score > best_score:
                best_score = score
                best_move = i
    return best_move

# Função para verificar o vencedor
def check_winner(board):
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Linhas
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Colunas
        [0, 4, 8], [2, 4, 6],             # Diagonais
    ]
    for line in win_conditions:
        if board[line[0]] == board[line[1]] == board[line[2]] and board[line[0]] != 0:
            return board[line[0]]  # Retorna 1 ou -1 para vitória
    if 0 not in board:  # Empate
        return 0
    return None  # Jogo ainda não acabou