# different cases of distances on the node 'table-0' of tables-2.html

DEBUG_THRESHOLD = 0.5
TOO_FAR = 0.9
CLOSE_ENOUGH = 0.1

case_0 = {
    1: {
        ((0, 1), (1, 2)): TOO_FAR,
        ((1, 2), (2, 3)): TOO_FAR,
        ((2, 3), (3, 4)): TOO_FAR,
        ((3, 4), (4, 5)): TOO_FAR,
        ((4, 5), (5, 6)): TOO_FAR,
        ((5, 6), (6, 7)): TOO_FAR,
        ((6, 7), (7, 8)): TOO_FAR,
        ((7, 8), (8, 9)): TOO_FAR,
        ((8, 9), (9, 10)): TOO_FAR,
        ((9, 10), (10, 11)): TOO_FAR,
        ((10, 11), (11, 12)): TOO_FAR,
    },
    2: {
        ((0, 2), (2, 4)): TOO_FAR,
        ((1, 3), (3, 5)): TOO_FAR,
        ((2, 4), (4, 6)): TOO_FAR,
        ((3, 5), (5, 7)): TOO_FAR,
        ((4, 6), (6, 8)): TOO_FAR,
        ((5, 7), (7, 9)): TOO_FAR,
        ((6, 8), (8, 10)): TOO_FAR,
        ((7, 9), (9, 11)): TOO_FAR,
        ((8, 10), (10, 12)): TOO_FAR,
    },
    3: {
        ((0, 3), (3, 6)): TOO_FAR,
        ((1, 4), (4, 7)): TOO_FAR,
        ((2, 5), (5, 8)): TOO_FAR,
        ((3, 6), (6, 9)): TOO_FAR,
        ((4, 7), (7, 10)): TOO_FAR,
        ((5, 8), (8, 11)): TOO_FAR,
        ((6, 9), (9, 12)): TOO_FAR,
    }
}

case_1 = {
    1: {
        ((0, 1), (1, 2)): CLOSE_ENOUGH,
        ((1, 2), (2, 3)): CLOSE_ENOUGH,
        ((2, 3), (3, 4)): TOO_FAR,
        ((3, 4), (4, 5)): CLOSE_ENOUGH,
        ((4, 5), (5, 6)): TOO_FAR,
        ((5, 6), (6, 7)): CLOSE_ENOUGH,
        ((6, 7), (7, 8)): TOO_FAR,
        ((7, 8), (8, 9)): TOO_FAR,
        ((8, 9), (9, 10)): CLOSE_ENOUGH,
        ((9, 10), (10, 11)): CLOSE_ENOUGH,
        ((10, 11), (11, 12)): TOO_FAR,
    },
    2: {
        ((0, 2), (2, 4)): TOO_FAR,
        ((2, 4), (4, 6)): CLOSE_ENOUGH,
        ((4, 6), (6, 8)): CLOSE_ENOUGH,
        ((6, 8), (8, 10)): CLOSE_ENOUGH,
        ((8, 10), (10, 12)): TOO_FAR,

        ((1, 3), (3, 5)): TOO_FAR,
        ((3, 5), (5, 7)): CLOSE_ENOUGH,
        ((5, 7), (7, 9)): CLOSE_ENOUGH,
        ((7, 9), (9, 11)): CLOSE_ENOUGH,
    },
    3: {
        ((0, 3), (3, 6)): CLOSE_ENOUGH,  # change here for next case
        ((3, 6), (6, 9)): CLOSE_ENOUGH,
        ((6, 9), (9, 12)): TOO_FAR,

        ((1, 4), (4, 7)): CLOSE_ENOUGH,
        ((4, 7), (7, 10)): CLOSE_ENOUGH,

        ((2, 5), (5, 8)): TOO_FAR,
        ((5, 8), (8, 11)): TOO_FAR,
    }
}



case_2 = {
    1: {
        ((0, 1), (1, 2)): CLOSE_ENOUGH,
        ((1, 2), (2, 3)): CLOSE_ENOUGH,
        ((2, 3), (3, 4)): TOO_FAR,
        ((3, 4), (4, 5)): CLOSE_ENOUGH,
        ((4, 5), (5, 6)): TOO_FAR,
        ((5, 6), (6, 7)): CLOSE_ENOUGH,
        ((6, 7), (7, 8)): TOO_FAR,
        ((7, 8), (8, 9)): TOO_FAR,
        ((8, 9), (9, 10)): CLOSE_ENOUGH,
        ((9, 10), (10, 11)): CLOSE_ENOUGH,
        ((10, 11), (11, 12)): TOO_FAR,
    },
    2: {
        ((0, 2), (2, 4)): TOO_FAR,
        ((2, 4), (4, 6)): CLOSE_ENOUGH,
        ((4, 6), (6, 8)): CLOSE_ENOUGH,
        ((6, 8), (8, 10)): CLOSE_ENOUGH,
        ((8, 10), (10, 12)): TOO_FAR,

        ((1, 3), (3, 5)): TOO_FAR,
        ((3, 5), (5, 7)): CLOSE_ENOUGH,
        ((5, 7), (7, 9)): CLOSE_ENOUGH,
        ((7, 9), (9, 11)): CLOSE_ENOUGH,
    },
    3: {
        ((0, 3), (3, 6)): TOO_FAR,  # change here for next case
        ((3, 6), (6, 9)): CLOSE_ENOUGH,
        ((6, 9), (9, 12)): TOO_FAR,

        ((1, 4), (4, 7)): CLOSE_ENOUGH,
        ((4, 7), (7, 10)): CLOSE_ENOUGH,

        ((2, 5), (5, 8)): TOO_FAR,
        ((5, 8), (8, 11)): TOO_FAR,
    }
}


all_cases = [var for var_name, var in globals().items() if var_name.startswith("case_")]
