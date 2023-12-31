import pygame
import cv2
import numpy as np
from keras.models import load_model
import sys
import os
import matplotlib.pyplot as plt
# load the trained model
model = load_model('mnist.keras')

def preprocessing_images(image_directory):
  img = cv2.imread(image_directory) # read the image
  img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert from RGB to grayscale
  resized_img = cv2.resize(img_grayscale, (28,28), interpolation=cv2.INTER_AREA ) # resized image
  resized_img = resized_img / 255.0 # normalize image
  resized_img = resized_img.reshape(-1,28,28,1) # expand the dimension of the image to be the same as the input shape
  return resized_img

def predicting_digit(model, image):
  result = model.predict(image)
  return f"The digit may be is {np.argmax(result)}, accuracy: {max(result[0]) * 100:.2f}"

def show_image(image):
    plt.imshow(image.reshape(28,28), cmap=plt.cm.binary)
    plt.show()

# setup
black = (0, 0, 0) # color in RGB
white = (255, 255, 255) # white in RGB
draw_on = False # drawing is active or not
last_pos = (0, 0) # last position the brush was
radius = 5 # radius of the Brush
# screen
width, height = 500, 500
screen = pygame.display.set_mode((width,height))
pygame.display.set_caption("Digit Recognizer")
# separate surface for drawing
drawing_surface = pygame.Surface((width, height-50))
drawing_surface.fill(black)



def roundline(canvas, color, start, end, radius=1):
    """
    draws a rounded line between two points
    :param canvas: the surface to draw on
    :param color: color in RGB
    :param start: starting position
    :param end: ending position
    :param radius: radius of circles
    :return:
    """
    x_axis = end[0] - start[0]
    y_axis = end[1] - start[1]
    dist = max(abs(x_axis), abs(y_axis))
    for i in range(dist):
        x = int(start[0] + float(i) / dist * x_axis)
        y = int(start[1] + float(i) / dist * y_axis)
        pygame.draw.circle(canvas, color, (x, y), radius)

def clear_screen():
    drawing_surface.fill(black)

def classify_digit(): # use to print to the stdout
    pygame.image.save(drawing_surface, "captured_drawing.png")
    captured_image_path = "captured_drawing.png"
    image = preprocessing_images(captured_image_path)
    print(predicting_digit(model, image))
    show_image(image)
    if os.path.exists(captured_image_path):
        os.remove(captured_image_path)

def classify_digit2(): # use to print to the screen
    pygame.image.save(drawing_surface, "captured_drawing.png")
    captured_image_path = "captured_drawing.png"
    image = preprocessing_images(captured_image_path)
    classification_result = predicting_digit(model, image)

    # create a message box surface
    font = pygame.font.Font(None, 36)
    text_surface = font.render(classification_result, True, white)
    text_rect = text_surface.get_rect(center=(width // 2, height - 100))
    # display the message box
    drawing_surface.fill(black)
    drawing_surface.blit(text_surface, text_rect)
    # update the display
    screen.blit(drawing_surface, (0, 0))
    pygame.display.flip()
    # wait for a moment before clearing the message box
    pygame.time.delay(5000)
    # clear the message box
    drawing_surface.fill(black)
    if os.path.exists(captured_image_path):
        os.remove(captured_image_path)


try:
    while True: # loop until the user quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # if user quits then the loop stop
                raise StopIteration

            if event.type == pygame.MOUSEBUTTONDOWN: # if the user do a single mouse click, draw a circle
                x, y = event.pos
                if 20 <= x <= 120 and height - 40 <= y <= height - 10:
                    clear_screen()
                # check if the mouse click is within the "Classify" button
                elif width - 210 <= x <= width - 80 and height - 40 <= y <= height - 10:
                    classify_digit2()
                else:
                    pygame.draw.circle(drawing_surface, white, event.pos, radius)
                    draw_on = True  # set drawing to active
            if event.type == pygame.MOUSEBUTTONUP: # when the mouse button is released, stop
                draw_on = False # set drawing to inactive

            if event.type == pygame.MOUSEMOTION: # draw a continuous circle when hold the mouse button
                if draw_on:
                    pygame.draw.circle(drawing_surface, white, event.pos, radius)
                    roundline(drawing_surface, white, event.pos, last_pos, radius) # create smoother lines
                last_pos = event.pos

        screen.blit(drawing_surface, (0,0))
        pygame.draw.rect(screen, (61, 130, 178), (20, height - 40, 100, 30))  # clear button
        pygame.draw.rect(screen, (61, 130, 178), (width - 210, height - 40, 130, 30))  # classify button

        # draw text
        pygame.font.init()
        font = pygame.font.Font(None,size = 36)
        text_clear = font.render("Clear", True, white)
        text_classify = font.render("Classify", True, white)
        screen.blit(text_clear, (40, height - 35))
        screen.blit(text_classify, (width - 190, height - 35))
        pygame.display.flip() # updates the display to reflect changes made during loop

except StopIteration:
    pass


# quit
pygame.quit()