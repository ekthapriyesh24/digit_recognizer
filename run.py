from PIL import Image
img = Image.open('img.jpg').convert('LA')
img.save('greyscale.png')
print("Hello")