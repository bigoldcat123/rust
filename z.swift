func minDistance(_ word1: String, _ word2: String) -> Int {
    let i = word1.count
    let j = word2.count
    var dp: [[Int]] = Array(repeating: Array(repeating: 0, count: j + 1), count: i + 1)
    for _i in 0...i {
        dp[_i][0] = _i
    }
    for _j in 0...j {
        dp[0][_j] = _j
    }
    if !(i == 0 || j == 0) {
        for _i in 1...i {
            for _j in 1...j {
                if word1.utf8CString[_i - 1] == word2.utf8CString[_j - 1] {
                    dp[_i][_j] =
                        min(dp[_i][_j - 1], min(dp[_i - 1][_j], dp[_i - 1][_j - 1] - 1)) + 1
                } else {
                    dp[_i][_j] = min(dp[_i][_j - 1], min(dp[_i - 1][_j], dp[_i - 1][_j - 1])) + 1
                }
            }
        }
    }

    // print(dp)
    for i in dp {
        print(i)
    }

    return dp[i][j]
}

func setZeroes(_ matrix: inout [[Int]]) {
    var xs: [Int] = [];
    var ys: [Int] = [];
    for i in matrix.enumerated() {
        for j in i.element.enumerated() {
            if j.element == 0 {
                xs.append(i.offset)
                ys.append(j.offset)
            }
        }
    }
        for i in matrix.enumerated() {
        for j in i.element.enumerated() {
            if xs.contains(i.offset) || ys.contains(j.offset) {
                matrix[i.offset][j.offset] = 0
            }
        }
    }
}
