import cv2
import numpy as np

rows = 6 
cols = 4
square_size = 50

board_height = (rows + 1) * square_size
board_width = (cols + 1) * square_size

chessboard_img = np.full((board_height, board_width), 255, np.uint8)

for row in range(rows+1):
    for col in range(cols + 1):
        if (row + col) % 2 == 0:
            x_start = col * square_size
            y_start = row * square_size
            x_end = x_start + square_size
            y_end = y_start + square_size
            cv2.rectangle(chessboard_img, (x_start, y_start), (x_end, y_end), 0, -1)
cv2.imwrite('chessboard.png', chessboard_img)
