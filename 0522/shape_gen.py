import matplotlib.pyplot as plt
import random

def ru():
    return random.uniform(0.001, 0.999)

def rc():
    return (ru(), ru(), ru())

for i in range(1000):
    shape_type= random.randint(0,2)
    filename = ""
    tri_num = 0
    circle_num = 0
    rect_num = 0
    if shape_type == 0:
        shape = plt.Polygon(([ru()/2, ru()],[ru()/2+0.5, ru()],[ru(), ru()]), color=rc())
        tri_num += 1
        filename = "triangle_" + str(tri_num) + "_" + str(i+1) + ".png"
    elif shape_type == 1:
        shape = plt.Circle((ru()/2+0.25, ru()/2+0.25), ru()/4, color=rc())
        circle_num += 1
        filename = "circle_" + str(tri_num) + "_" + str(i+1) + ".png"
    else:
        shape = plt.Rectangle((ru(), ru()), ru(), ru(), fc=rc())
        rect_num += 1
        filename = "rectangle_" + str(tri_num) + "_" + str(i+1) + ".png"

    fig, ax = plt.subplots()
    fig.set_size_inches(1,1)
    plt.axis('off')

    ax.add_artist(shape)
    fig.savefig('Shape/'+filename)
