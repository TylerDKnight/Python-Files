def generatePascal(depth):
    p = {0:[1]}
    for row in range(1,depth):
        thisRow = [0]*(row+1)
        for column in range(len(thisRow)):
            if column == 0 or column == len(thisRow)-1:
                thisRow[column] = 1
            else:
                thisRow[column] = p[row-1][column] + p[row-1][column-1]
        p[row] = thisRow
    return p

def simpleDisplayPascal(depth):
  triangle = generatePascal(depth)
  for row in list(triangle.keys()):
    rowString = ''
    for num in triangle[row]:
      rowString += str(num) + " "
    print(rowString)

def displayPascal(depth):
    triangle = generatePascal(depth)
    displayString = ''
    length = max(triangle.keys(), key=int)
    for row in range(length,-1,-1):
        displayRow = ''
        for number in triangle[row]:
            displayRow += str(number) + ' '
        displayRow = displayRow[:-1]
        if row == length:
            maxLength = len(displayRow)
        padding = (maxLength-len(displayRow))//2*' '
        displayRow = padding + displayRow[::-1] + padding
        displayString += displayRow + '\n'
    return displayString[::-1]

print("Pascal(10):")
simpleDisplayPascal(10)
