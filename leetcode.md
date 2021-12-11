# LeetCode <<剑指offer>>

## 03 数组的重复的数字

```java
找出数组中重复的数字。

在一个长度为 n 的数组 nums 里的所有数字都在 0 ～ n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。
    
方式一： 原地hash
class Solution {
    public int findRepeatNumber(int[] nums) {
        for (int i = 0, n = nums.length; i < n; ++i) {
            while (nums[i] != i) {
                if (nums[i] == nums[nums[i]]) return nums[i];
                swap(nums, i, nums[i]);
            }
        }
        return -1;
    }

    private void swap(int[] nums, int i, int j) {
        int t = nums[i];
        nums[i] = nums[j];
        nums[j] = t;
    }
}

方式二： 排序
		Arrays.sort(nums);
        for (int i = 0; i < nums.length; i ++ ) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                return nums[i];
            }
        }
        return -1;

方式三： map
        Map<Integer, Integer> mp = new HashMap<>();
        for (int i = 0; i < nums.length; i ++ ) {
            mp.put(nums[i], mp.getOrDefault(nums[i], 0) + 1);
        }
        for (int i = 0; i < nums.length; i ++ ) {
            if (mp.get(nums[i]) > 1) {
                return nums[i];
            }
        }
        return -1;
```



## 04 二维数组中的查找

```java
在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

重点：固定一维最大，另一维最小
    
public boolean findNumberIn2DArray(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
            	return false;

        int m = matrix.length;
        int n = matrix[0].length;

        int i = 0, j = n - 1;
        while (i < m && j >= 0) {
            if (matrix[i][j] == target) return true;
            else if (matrix[i][j] > target) {
                j--;
            } else {
                i++;
            }
        }
        return false;
    }
```



## 05 替换空格

```java
请实现一个函数，把字符串 s 中的每个空格替换成"%20"。

public String replaceSpace(String ss) {
        char[] s = ss.toCharArray();
        StringBuilder str = new StringBuilder();
        for (int i = 0; i < s.length; i ++ ) {
            if (s[i] == ' ') {
                str.append("%20");
            } else {
                str.append(s[i]);
            }
        }
        return str.toString();
    }
```



## 12 矩阵中的路径

```java
请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如，在下面的 3×4 的矩阵中包含一条字符串“bfce”的路径（路径中的字母用加粗标出）。

// 回溯法 + 上下左右方向搜索
    
class Solution {
    private boolean[][] vis;
    int[] X = {0, 0, 1, -1};
    int[] Y = {1, -1, 0, 0};
    public boolean exist(char[][] board, String word) {
        int m = board.length;
        int n = board[0].length;
        vis = new boolean[m][n];
        char[] s = word.toCharArray();
        for (int i = 0; i < m; i ++ ) {
            for (int j = 0; j < n; j ++ ) {
                if (dfs(board, i, j, s, 0) == true) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean dfs(char[][] board, int x, int y, char[] s, int idx) {
        if (idx == s.length) return true;
        if (x < 0 || x >= board.length || y < 0 || y >= board[0].length) 
            	return false;
        if (vis[x][y] || s[idx] != board[x][y]) return false;

        vis[x][y] = true;
        boolean res = false;   
        for (int i = 0; i < 4; i ++) {
            int nx = X[i] + x;
            int ny = Y[i] + y;
            if (dfs(board, nx, ny, s, idx + 1) == true) {
                res = true;
                break;
            }
        }
        // res = dfs(board, x + 1, y, s, idx + 1) || dfs(board, x - 1, y, s, idx + 1) || 
        //     dfs(board, x, y + 1, s, idx + 1) || dfs(board, x, y - 1, s, idx + 1);

        vis[x][y] = false;

        return res;
    }
}
```



## 19 正则表达式匹配

```java
```





## 20 表示数值的字符串

```java
请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100"、"5e2"、"-123"、"3.1416"、"0123"及"-1E-16"都表示数值，但"12e"、"1a3.14"、"1.2.3"、"+-5"及"12e+5.4"都不是。
/*
关键点： 反证法
出现 +/- 时，位置必须是在第 0 位，或者 e/E 的后面一位
出现 . 时，在此之前不能出现 . 或者 e/E
出现 e/E 时，前面不能出现 e/E，并且必须出现过数字
*/
class Solution {
    public boolean isNumber(String ss) {
        if (ss == null || ss.trim().length() == 0) return false;

        char[] s = ss.trim().toCharArray();
        boolean hasNum = false;
        boolean hasDot = false;
        boolean hasE = false;
        for (int i = 0; i < s.length; i ++ ) {
            if (s[i] >= '0' && s[i] <= '9') {
                hasNum = true;
            } else if (s[i] == '+' || s[i] == '-') {
                if (i > 0 && !(s[i - 1] == 'E' || s[i - 1] == 'e')) {
                    return false;
                }
            } else if (s[i] == 'E' || s[i] == 'e') {
                if (hasE || !hasNum) {
                    return false;
                }
                hasE = true;
                hasNum = false;
            } else if (s[i] == '.') {
                if (hasDot || hasE) {
                    return false;
                }
                hasDot = true;
            } else {
                return false;
            }
        }
        return hasNum;
    }
}
```





## 21 调整数组顺序使奇数位于偶数前面

```java
输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。

//方式一： 开辟新数组    方式二： 类似快速排序的方式
public int[] exchange(int[] nums) {
        // int[] res = new int[nums.length];
        // int idx = 0;
        // for (int i = 0; i < nums.length; i ++ ) {
        //     if (nums[i] % 2 != 0) {
        //         res[idx++] = nums[i];
        //     }
        // }
        // for (int i = 0; i < nums.length; i ++ ) {
        //     if (nums[i] % 2 == 0) {
        //         res[idx++] = nums[i];
        //     }
        // }
        // return res;

        int i = 0, j = nums.length - 1;
        while (i < j) {
            while (i < j && nums[i] % 2 != 0) i++;
            while (i < j && nums[j] % 2 == 0) j--;
            if (i <= j) {
                swap(nums, i, j);
            }
        }
        return nums;
    }
    private void swap(int[] nums, int i, int j) {
        int t = nums[i];
        nums[i] = nums[j];
        nums[j] = t;
    }
```



# Carl习题集

