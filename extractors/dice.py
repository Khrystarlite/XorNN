from __future__ import print_function
import random

min = 1
DB = 5
roll_again = "yes"
def Line(x):
    return {
        1:	428,
        2:	423,
        3:	312,
        4:	31,
        5:	382,
    }[x]

while roll_again == "yes" or roll_again == "YES" or roll_again == "y" or roll_again == "Y" or roll_again == '1':
    print ("Rolling the dice...")
    db = random.randint(min, DB)
    print ("DB: ", db)
    line = Line(db)
    print ("Lines: ", random.randint(min, line))

    roll_again = raw_input("Roll the dices again? ")
    
del min, line, DB, roll_again