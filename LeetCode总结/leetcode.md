![](.\pic\leetcode\刷题大纲.webp)

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

## 06 从尾到头打印链表

```java
输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。
// 方式一： 求全长，倒着放在数组中 
// 方式二:  使用栈    
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public int[] reversePrint(ListNode head) {
        // Stack<Integer> st = new Stack<>();
        // while (head != null) {
        //     st.push(head.val);
        //     head = head.next;
        // }
        // int idx = 0;
        // int[] res = new int[st.size()];
        // while (!st.isEmpty()) {
        //     res[idx++] = st.pop();
        // }
        // return res;

        ListNode cur = head;
        int n = 0;
        while (cur != null) {
            cur = cur.next;
            n++;
        }

        int idx = n - 1;
        int[] res = new int[n];
        while (head != null) {
            res[idx--] = head.val;
            head = head.next;
        }
        return res;
    }
}
```

## 07 重建二叉树

```java
题目一： 从前序与中序遍历序列构造二叉树 lc105
前： [1] 2  k k + 1 n
中:  1 k - 1  [k] k + 1  n
// 优化： 中序数字的下标用hash存储
class Solution {
    Map<Integer, Integer> mp = new HashMap<>();
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        for (int i = 0; i < inorder.length; i ++ ) {
            mp.put(inorder[i], i);
        }
        return create(preorder, inorder, 0, preorder.length - 1, 0, inorder.length - 1);
    }

    private TreeNode create(int[] preorder, int[] inorder, int preL, int preR, int inL, int inR) {
        if (preL > preR) return null;

        int root = preorder[preL];
        int k = mp.get(root);
        // for (k = inL; k <= inR; k ++ ) {
        //     if (root == inorder[k]) {
        //         break;
        //     }
        // }
        int numL = k - inL;

        TreeNode node = new TreeNode(root);
        node.left = create(preorder, inorder, preL + 1, preL + numL, inL, k - 1);
        node.right = create(preorder, inorder, preL + numL + 1, preR, k + 1, inR);

        return node;
    }
}

题目二： 从后序与中序遍历序列构造二叉树  lc106
后： 1   k - 1 k  n - 1  n
中:  1 k - 1  [k] k + 1  n 
// 优化： 中序数字的下标用hash存储
class Solution {
    Map<Integer, Integer> mp = new HashMap<>();
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        for (int i = 0; i < inorder.length; i ++ ) {
            mp.put(inorder[i], i);
        }
        return create(postorder, inorder, 0, postorder.length - 1, 0, inorder.length - 1);
    }

    private TreeNode create(int[] postorder, int[] inorder, int postL, int postR, int inL, int inR) {
        if (postL > postR) return null;

        int root = postorder[postR];
        int k = mp.get(root);
        // for (k = inL; k <= inR; k ++ ) {
        //     if (root == inorder[k]) {
        //         break;
        //     }
        // }
        int numL = k - inL;
        
        TreeNode node = new TreeNode(root);
        node.left = create(postorder, inorder, postL, postL + numL - 1, inL, k - 1);
        node.right = create(postorder, inorder, postL + numL, postR - 1, k + 1, inR);

        return node; 
    }
}
```

## 09 用两个栈实现队列

```java
用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )

class CQueue {
    private Stack<Integer> st1;
    private Stack<Integer> st2;

    public CQueue() {
        st1 = new Stack<>();
        st2 = new Stack<>();
    }
    
    public void appendTail(int value) {
        st1.push(value);
    }
    
    public int deleteHead() {
        if (st2.isEmpty()) {
            while (!st1.isEmpty()) {
                st2.push(st1.pop());
            }
        }
        if (st2.isEmpty()) {
            return -1;
        } else {
            int res = st2.pop();
            return res;
        }
    }
}
```

## 10-I  斐波那契数列

```java
F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。
// 注意： return a 且 F[0] = 0   F[1] = 1  
class Solution {
    public int fib(int n) {
        // if (n == 0) return n;
        // if (n == 1) return n;

        // int[] f = new int[n + 1];
        // f[0] = 0;
        // f[1] = 1;
        // for (int i = 2; i <= n; i ++ ) {
        //     f[i] = (int) (f[i - 1] + f[i - 2]) % 1000000007;
        // }
        // return f[n];

        if (n == 0) return 0;
        if (n == 1) return 1;

        int a = 0, b = 1;
        for (int i = 0; i < n; i ++ ) {
            int sum = (a + b) % 1000000007;
            a = b;
            b = sum;
        }
        return a;
    }
}
```

## 10-II 青蛙跳台阶问题

```java
一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。


// 注意： return b 且 F[0] = F[1] = 1
class Solution {
    public int numWays(int n) {
        // if(n == 0 || n == 1) return 1;

        // int[] f = new int[n + 1];
        // f[0] = f[1] = 1;
        // for (int i = 2; i <= n; i ++ ) {
        //     f[i] = (int)((f[i - 1] + f[i - 2]) % 1000000007);
        // }
        // return f[n];


        if (n == 0 || n == 1) return 1;
        int a = 0, b = 1;
        for (int i = 0; i < n; i ++ ) {
            int sum = (a + b) % 1000000007;
            a = b;
            b = sum;
        }
        return b;
    }
}
```

## 11 旋转数组的最小数字

```java
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组  [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为 1。
// 二分法 + 去除重复
class Solution {
    public int minArray(int[] A) {
        int l = 0, r = A.length - 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (A[mid] > A[r]) {
                l = mid + 1;
            } else if (A[mid] < A[r]) {
                r = mid;
            } else {
                r--;
            }
        }
        return A[l];
    }
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

## 13 机器人的运动范围

```java
地上有一个 m 行 n 列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于 k 的格子。例如，当 k 为 18 时，机器人能够进入方格 [35, 37] ，因为 3+5+3+7=18。但它不能进入方格 [35, 38]，因为 3+5+3+8=19。请问该机器人能够到达多少个格子？

// dfs 
class Solution {
    private int m;
    private int n;
    private int k;
    private int cnt;
    private boolean[][] vis;
    private int[] X = {0, 0, 1, -1};
    private int[] Y = {1, -1, 0, 0};
    public int movingCount(int m, int n, int k) {
        this.m = m;
        this.n = n;
        this.k = k;
        cnt = 0;
        vis = new boolean[m][n];
        dfs(0, 0);
        return cnt;
    }

    private void dfs(int x, int y) {
        if (x < 0 || x >= m || y < 0 || y >= n) return;
        if (vis[x][y] || getSum(x) + getSum(y) > k) return;
        cnt++;
        vis[x][y] = true;
        for (int i = 0; i < 4; i ++ ) {
            int nx = X[i] + x;
            int ny = Y[i] + y;
            dfs(nx, ny);
        }
    }

    private int getSum(int n) {
        int res = 0;
        while (n != 0) {
            res += n % 10;
            n /= 10;
        }
        return res;
    }
}
```

## 14-I 剪绳子

```java
给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n 都是整数，n>1 并且 m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是 8 时，我们把它剪成长度分别为 2、3、3 的三段，此时得到的最大乘积是 18。

// 尽可能将绳子以长度 3 等分剪为多段时，乘积最大。    
class Solution {
    public int cuttingRope(int n) {
        // if (n == 1) return 1;
        // if (n == 2) return 1;
        // if (n == 3) return 2;

        // int[] f = new int[n + 1];
        // f[1] = 1;
        // f[2] = 2;
        // f[3] = 3;
        // for (int i = 4; i <= n; i ++ ) {
        //     for (int j = 1; j <= i / 2; j ++ ) {
        //         f[i] = Math.max(f[i], f[j] * f[i - j]);
        //     }
        // }
        // return f[n];

        if (n == 1) return 1;
        if (n == 2) return 1;
        if (n == 3) return 2;

        int res = 1;
        while (n > 4) {
            res *= 3;
            n -= 3;
        }
        return res * n;
    }
}
```

## 14-ii 剪绳子

```java
给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m - 1] 。请问 k[0]*k[1]*...*k[m - 1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

class Solution {
    public int cuttingRope(int n) {
        if (n == 1) return 1;
        if (n == 2) return 1;
        if (n == 3) return 2;
        long res = 1;
        while (n > 4) {
            res = (res * 3) % 1000000007;
            n -= 3;
        }
        return (int)((res * n) % 1000000007);
    }
}
```

## 15 二进制中1的个数

```java
// java
// n & (n - 1) 会消除 n 中最后一位中的 1。
public class Solution {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int cnt = 0; 
        while (n != 0) {
            n &= (n - 1);
            cnt++;
        }
        return cnt;
    }
}

// c++
class Solution {
public:
    int hammingWeight(uint32_t n) {
        uint32_t flag = 1;
        int res = 0;
        while (flag) {
            if (flag & n) res ++;
            flag <<= 1;
        }
        return res;
    }
};
```

## 16 数值的整数次方

```java
实现函数 double Power(double base, int exponent)，求 base 的 exponent 次方。不得使用库函数，同时不需要考虑大数问题。
    
public double myPow(double x, int n) {
    if (n < 0) {
        n = -n;
        x = 1.0 / x;
    }
    double res = 1.0;
    while (n != 0) {
        if (n % 2 != 0) res *= x;
        x = x * x;
        n /= 2;
    }
    return res;
}
```

## 17 打印从1到最大的n位数

```java
输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。
    
class Solution {
    public int[] printNumbers(int n) {
        int num = (int) Math.pow(10, n) - 1;
        int[] res = new int[num];
        for (int i = 0; i < num; i ++ ) {
            res[i] = i + 1;
        }
        return res;
    }
}
```

## 18 删除链表的节点

```java
给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。
返回删除后的链表的头节点。

// 新头节点 + head（在原链表操作）
// 新头节点 + 原链表节点（符合条件）
    
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode deleteNode(ListNode head, int val) {
        ListNode dummyNode = new ListNode(0);
        dummyNode.next = head;
        ListNode pre = dummyNode;

        while (pre.next != null && pre.next.val != val) {
            pre = pre.next;
        }

        if (pre.next == null) {
            pre.next = null;
        } else {
            pre.next = pre.next.next;
        }

        return dummyNode.next;
    }
}
```

## 19 正则表达式匹配

```java
请实现一个函数用来匹配包含'. '和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（含 0 次）。在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但与"aa.a"和"ab*a"均不匹配。

/*
情况一
f[i][j] = f[i][j] || f[i - 1][j - 1]
aaa
ba.
  a

情况二
f[i][j] = f[i][j] || f[i][j - 2](.* or a*表示空)
aaa
ba*
b.*

f[i][j] = f[i][j] || f[i - 1][j](.* or a*表示非空)
*/
    
class Solution {
    public boolean isMatch(String ss, String pp) {
        char[] s = ss.toCharArray();
        char[] p = pp.toCharArray();
        boolean[][] f = new boolean[s.length + 1][p.length + 1];
        for (int i = 0; i <= s.length; i ++ ) {
            for (int j = 0; j <= p.length; j ++ ) {
                if (i == 0 && j == 0) {
                    f[i][j] = true;
                    continue;
                }
                if (i != 0 && j == 0) {
                    f[i][j] = false;
                    continue;
                }
                f[i][j] = false;
                if (p[j - 1] != '*') {
                    if (i >= 1 && (p[j - 1] == '.' || p[j - 1] == s[i - 1])) {
                        f[i][j] = f[i][j] || f[i - 1][j - 1];
                    }
                } else {
                    if (j >= 2) f[i][j] = f[i][j] || f[i][j - 2];
                    if (j >= 2 && i >= 1 && 
                        (p[j - 2] == '.' || p[j - 2] == s[i - 1])) {
                        f[i][j] = f[i][j] || f[i - 1][j];
                    }
                }
            } 
        }
        return f[s.length][p.length];
    }
}
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

## 22 链表中倒数第k个节点

```java
输入一个链表，输出该链表中倒数第 k 个节点。为了符合大多数人的习惯，本题从 1 开始计数，即链表的尾节点是倒数第 1 个节点。例如，一个链表有 6 个节点，从头节点开始，它们的值依次是 1、2、3、4、5、6。这个链表的倒数第 3 个节点是值为 4 的节点。
    
// 快慢指针    
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode getKthFromEnd(ListNode head, int k) {
        ListNode slow = head;
        ListNode fast = head;
        while (k-- > 0) {
            fast = fast.next;
        } 
        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        } 
        return slow;
    }
}
```

## 24 翻转链表

```java
class Solution {
    public ListNode reverseList(ListNode head) {
        // 方式一: 迭代法
        // ListNode pre = null;
        // ListNode cur = head;
        // while (cur != null) {
        //     ListNode tmp = cur.next;
        //     cur.next = pre;
        //     pre = cur;
        //     cur = tmp;
        // }
        // return pre;

        // 方式二： 递归法
        if (head == null || head.next == null) return head;
        ListNode nHead = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return nHead;  
    }
}
```

## 25 合并两个排序的链表

```java
// 归并排序的思路
// 注意剩下链表的链接

class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummyNode = new ListNode(0);
        ListNode cur = dummyNode;

        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                cur.next = l1;
                l1 = l1.next;
            } else {
                cur.next = l2;
                l2 = l2.next;
            }
            cur = cur.next;
        }
        cur.next = l1 != null ? l1 : l2;
        return dummyNode.next;
    }
```

## 26 树的子结构

```java
// 先判断是否相等，然后左子树是否存在，最后右子树是否存在

class Solution {
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        if (B == null || A == null) return false;
        return isSubStructure(A.left, B) || isSubStructure(A.right, B) || check(A, B);    
    }

    private check(TreeNode A, TreeNode B) { // 判断是否相等
        if (B == null) return true;
        if (A == null && B != null) return false;
        if (A.val != B.val) return false; 

        return check(A.left, B.left) && check(A.right, B.right);
    }
}
```

## 27 二叉树的镜像

```java
//请完成一个函数，输入一个二叉树，该函数输出它的镜像。
// 先序遍历
class Solution {
    public TreeNode mirrorTree(TreeNode root) {
        if (root == null) return null;

        TreeNode tmp =  root.left;
        root.left = root.right;
        root.right = tmp;

        mirrorTree(root.left);
        mirrorTree(root.right);

        return root;
    }
}
```

## 28 对称的二叉树

```java
class Solution {
    public boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        return isSymmetric(root.left, root.right);
    }

    private boolean isSymmetric(TreeNode left, TreeNode right) {
        if (left == null && right == null) return true;
        if (left != null && right == null) return false;
        if (left == null && right != null) return false;
        if (left.val != right.val) return false;

        return isSymmetric(left.left, right.right) && isSymmetric(left.right, right.left);
    }
}
```

## 29 顺时针打印矩阵

```java
// 注意不变的量，控制变量

class Solution {
    public int[] spiralOrder(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) return new int[]{};

        int m = matrix.length;
        int n = matrix[0].length;
        int[] res = new int[m * n];
        int idx = 0;
        int t = 0, b = m - 1;
        int l = 0, r = n - 1;
        while (true) {
            for (int i = l; i <= r; i ++ ) {
                res[idx++] = matrix[t][i];
            }
            if (++t > b) break;

            for (int i = t; i <= b; i ++ ) {
                res[idx++] = matrix[i][r];
            }
            if (--r < l) break;

            for (int i = r; i >= l; i -- ) {
                res[idx++] = matrix[b][i];
            }
            if (--b < t) break;

            for (int i = b; i >= t; i -- ) {
                res[idx++] = matrix[i][l];
            }
            if (++l > r) break;
        }
        return res;
    }
}
```

## 30 包含min函数的栈

```java
定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。

class MinStack {
    private Stack<Integer> data;
    private Stack<Integer> minSt;

    /** initialize your data structure here. */
    public MinStack() {
        data = new Stack<>();
        minSt = new Stack<>();
    }
    
    public void push(int x) {
        data.push(x);
        if (minSt.isEmpty() || x < minSt.peek()) {
            minSt.push(x);
        } else {
            minSt.push(minSt.peek());
        }
    }
    
    public void pop() {
        data.pop();
        minSt.pop();
    }
    
    public int top() {
        return data.peek();
    }
    
    public int min() {
        return minSt.peek();
    }
}
```

## 31 栈的压入、弹出序列

```java
// 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。

// 直接使用栈模拟，如果栈顶等于弹出序列就弹出
class Solution {
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Stack<Integer> st = new Stack<>();
        int j = 0;
        for (int i = 0; i < pushed.length; i ++ ) {
            st.push(pushed[i]);
            while (!st.isEmpty() && st.peek() == popped[j]) {
                st.pop();
                j++;
            }
        }
        return st.isEmpty();
    }
}
```

## 32-i 从上到下打印二叉树

```java
从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。结果返回：[3,9,20,15,7]

// 层次遍历，注意返回的是int[]数组，最后需要转化
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public int[] levelOrder(TreeNode root) {
        if (root == null) return new int[0];
        Queue<TreeNode> q = new LinkedList<>();
        List<Integer> res = new ArrayList<>();
        q.add(root);
        while (!q.isEmpty()) {
            int n = q.size();
            for (int i = 0; i < n; i ++ ) {
                TreeNode node = q.poll();
                res.add(node.val);
                if (node.left != null) q.offer(node.left);
                if (node.right != null) q.offer(node.right);
            }
        }
        int[] ans = new int[res.size()];
        for (int i = 0; i < res.size(); i ++ ) {
            ans[i] = res.get(i);
        }
        return ans;
    }
}
```

## 32-ii 从上到下打印二叉树ii

```java
从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。

/*
结果返回
[
  [3],
  [9,20],
  [15,7]
]
*/
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);
        while (!q.isEmpty()) {
            int n = q.size();
            List<Integer> list = new ArrayList<>();
            for (int i = 0; i < n; i ++ ) {
                TreeNode node = q.poll();
                list.add(node.val);
                if (node.left != null) q.offer(node.left);
                if (node.right != null) q.offer(node.right);
            }
            res.add(list);
        }
        return res;
    }
}
```

## 32-iii 从上到下打印二叉树

```java
请实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。
    
// 奇数行不变，偶数行逆转(Collections.reverse(list))

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        Queue<TreeNode> q = new LinkedList<>();
        int dep = 0;
        q.add(root);
        while (!q.isEmpty()) {
            int n = q.size();
            List<Integer> row = new ArrayList<>();
            dep++;
            for (int i = 0; i < n; i ++ ) {
                TreeNode node = q.poll();
                row.add(node.val);
                if (node.left != null) q.add(node.left);
                if (node.right != null) q.add(node.right);
            }
            if (dep % 2 == 0) {
                Collections.reverse(row);
                res.add(row);
            } else {
                res.add(row);
            }
        }
        return res;
    }
} 
```

## 33 二叉搜索树的后序遍历序列

```java
//输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回  true，否则返回  false。假设输入的数组的任意两个数字都互不相同。
// 关键点： 左 < 中 < 右
class Solution {
    public boolean verifyPostorder(int[] postorder) {
        return dfs(postorder, 0, postorder.length - 1);
    }

    private boolean dfs(int[] postorder, int l, int r) {
        if (l > r) return true;
        int i = l;
        while (postorder[i] < postorder[r]) {
            i++;
        } 
        for (int j = i; j <= r - 1; j ++ ) {
            if (postorder[j] < postorder[r]) {
                return false;
            }
        }
        return dfs(postorder, l, i - 1) && dfs(postorder, i, r - 1);
    }
}
```

## 34 二叉树中和为某一值得路径 （路径总和II）lc113

```java
输入一棵二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。从树的根节点开始往下一直到叶节点所经过的节点形成一条路径。

class Solution {
    List<List<Integer>> res;
    LinkedList<Integer> path;
    public List<List<Integer>> pathSum(TreeNode root, int target) {
        res = new ArrayList<>();
        path = new LinkedList<>();
        if (root == null) return res;
        path.add(root.val);
        dfs(root, target - root.val);
        return res;
    }
    private void dfs(TreeNode root, int sum) {
        if (root == null) return;
        if (root.left == null && root.right == null && sum == 0) {
            res.add(new ArrayList<>(path));
            return;
        }
        if (root.left != null) {
            path.add(root.left.val);
            sum -= root.left.val;
            dfs(root.left, sum);
            sum += root.left.val;
            path.removeLast();
        }
        if (root.right != null) {
            path.add(root.right.val);
            sum -= root.right.val;
            dfs(root.right, sum);
            sum += root.right.val;
            path.removeLast();
        }
    }
}
```

## 34 路径总和I   lc112

```java
给你二叉树的根节点 root 和一个表示目标和的整数 targetSum 。判断该树中是否存在 根节点到叶子节点 的路径，这条路径上所有节点值相加等于目标和 targetSum 。如果存在，返回 true ；否则，返回 false 。

叶子节点 是指没有子节点的节点。

class Solution {
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) return false;
        return dfs(root, targetSum - root.val);    
    }
    private boolean dfs(TreeNode root, int sum) {
        if (root.left == null && root.right == null && sum == 0) return true; 
        if (root.left != null) {
            sum -= root.left.val;
            if (dfs(root.left, sum) == true) {
                return true;
            }
            sum += root.left.val;
        }
        if (root.right != null) {
            sum -= root.right.val;
            if (dfs(root.right, sum) == true) {
                return true;
            }
            sum += root.right.val;
        }
        return false;
    }
}
```

## 35 复杂链表的复制

```java
请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。

// 方法一：hash法构建新链表
class Solution {
    public Node copyRandomList(Node head) {
        Map<Node, Node> mp = new HashMap<>();
        for (Node p = head; p != null; p = p.next ) {
            mp.put(p, new Node(p.val));
        }
        for (Node p = head; p != null; p = p.next ) {
            mp.get(p).next = mp.get(p.next);
            mp.get(p).random = mp.get(p.random);
        }
        return mp.get(head);
    }
}
// 方法二：指针方法
class Solution {
    public Node copy(Node head) {
        if (head == null) return null;
        // 复制原始值
        Node cur = head;
        while (cur != null) {
            Node node = new Node(cur.val);
            node.next = cur.next;
            cur.next = node;
            cur = cur.next.next;
        }
        // 复制随机指针
        cur = head;
        while (cur != null) {
            cur.next.random = cur.random == null ? null : cur.random.next;
            cur = cur.next.next;
        }
        // 分裂成两个链表
        cur = head;
        Node copy = head.next;
        while (cur != null) {
            Node tmp = cur.next;
            cur.next = tmp.next;
            tmp.next = tmp.next == null ? null : tmp.next.next;
            cur = cur.next;
        }
        return copy;
    }
}
```

## 36 二叉搜索树与双向链表

```java
/* 双向链表：pre.right = cur、cur.left = pre、pre = cur

// Definition for a Node.
class Node {
    public int val;
    public Node left;
    public Node right;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val,Node _left,Node _right) {
        val = _val;
        left = _left;
        right = _right;
    }
};
*/
class Solution {
    private Node pre;
    private Node head;
    public Node treeToDoublyList(Node root) {
        if (root == null) return root;
        dfs(root);
        head.left = pre;
        pre.right = head;
        return head;
    }
    private void dfs(Node cur) {
        if (cur == null) return;
        dfs(cur.left);
        if (pre == null) {
            head = cur;
        } else {
            pre.right = cur;
        }
        cur.left = pre;
        pre = cur;
        dfs(cur.right);
    }
}
```

## 37 序列化二叉树

```java
你可以将以下二叉树：

    1
   / \
  2   3
     / \
    4   5

序列化为 "[1,2,3,null,null,4,5]"
    

public class Codec {
    public String serialize(TreeNode root) {
        if (root == null) return "[]";

        StringBuilder str = new StringBuilder("[");
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int n = queue.size();
            for (int i = 0; i < n; i ++ ) {
                TreeNode node = queue.poll();
                if (node != null) {
                    str.append(node.val);
                    queue.offer(node.left);
                    queue.offer(node.right);
                } else {
                    str.append("null");
                }
                str.append(",");
            }
        }
        return str.deleteCharAt(str.length() - 1).append("]").toString();
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if (data == null || data == "[]") return null;

        String[] nodes = data.substring(1, data.length() - 1).split(",");
        TreeNode root = new TreeNode(Integer.parseInt(nodes[0]));
        int idx = 1;
        Queue<TreeNode> queue = new LinkedList();
        queue.offer(root);
        while (!queue.isEmpty() && idx < nodes.length) {
            int n = queue.size();
            for (int i = 0; i < n; i ++ ) {
                TreeNode node = queue.poll();
                if (!"null".equals(nodes[idx])) {
                    node.left = new TreeNode(Integer.parseInt(nodes[idx]));
                    queue.offer(node.left);
                }
                idx++;
                if (!"null".equals(nodes[idx])) {
                    node.right = new TreeNode(Integer.parseInt(nodes[idx]));
                    queue.offer(node.right);
                }
                idx++;
            }
        }
        return root;
    }
}
```

## 38 字符串的排列

```java
// 输入一个字符串，打印出该字符串中字符的所有排列。
// 你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。

/*
输入：s = "abc"
输出：["abc","acb","bac","bca","cab","cba"]
*/

class Solution {
    private StringBuilder str;
    private List<String> res;
    private boolean[] used;
    public String[] permutation(String ss) {
        // 1、排序 2、去重 3、从0开始(不是组合，是排列)
        char[] s = ss.toCharArray();
        used = new boolean[s.length];
        str = new StringBuilder();
        res = new ArrayList<>();
        // 方式一: sort + used判断 去重
        // Arrays.sort(s);
        // backtracking(s, 0);
        // 方式二: set 去重
        backtracking2(s, 0);
        return res.toArray(new String[res.size()]);
    }
    // private void backtracking(char[] s, int idx) {
    //     if (idx == s.length) {
    //         res.add(str.toString());
    //         return;
    //     }
    //     for (int i = 0; i < s.length; i ++ ) {
    //         if (used[i] == true || (i > 0 && s[i] == s[i - 1] && used[i - 1] == false)) continue;
    //         used[i] = true;
    //         str.append(s[i]);
    //         backtracking(s, idx + 1);
    //         str.deleteCharAt(str.length() - 1);
    //         used[i] = false;
    //     }
    // }

    private void backtracking2(char[] s, int idx) {
        if (idx == s.length) {
            // res.add(new String(s));
            res.add(String.valueOf(s));
            return;
        }
        Set<Character> set = new HashSet<>(); 
        for (int i = idx; i < s.length; i ++ ) {
            if (set.contains(s[i])) continue;
            set.add(s[i]);
            swap(s, idx, i);
            backtracking2(s, idx + 1);
            swap(s, idx, i);
        }
    }
    private void swap(char[] s, int i, int j) {
        char t = s[i];
        s[i] = s[j];
        s[j] = t;
    }
}
```

## 39 数组中出现次数超过一半的数字

```java
数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
你可以假设数组是非空的，并且给定的数组总是存在多数元素。
    
class Solution {
    public int majorityElement(int[] nums) {
        // 投票法
        // int res = 0;
        // int cnt = 0;
        // for (int i = 0; i < nums.length; i ++ ) {
        //     if (cnt == 0) {
        //         res = nums[i];
        //         cnt++;
        //     } else {
        //         if (res == nums[i]) cnt++;
        //         else cnt--;
        //     }
        // }
        // return res;
        int res = nums[0];
        Map<Integer, Integer> mp = new HashMap<>();
        for (int i = 0; i < nums.length; i ++ ) {
            mp.put(nums[i], mp.getOrDefault(nums[i], 0) + 1);
            if (mp.getOrDefault(nums[i], 0) > nums.length / 2) {
                res = nums[i]; 
            }
        } 
        return res;
    }
}
```

## 40 最小的k个数

```java
class Solution {
    // 方式一： 优先队列，大顶堆
    public int[] getLeastNumbers(int[] arr, int k) {
        if (k == 0) return new int[0];
        // PriorityQueue<Integer> queue = new PriorityQueue<>((a, b) -> b - a);
        PriorityQueue<Integer> queue = new PriorityQueue<>(Collections.reverseOrder());
        for (int i = 0; i < arr.length; i ++ ) {
            if (queue.size() < k) {
                queue.offer(arr[i]);
            } else {
                if (queue.peek() > arr[i]) {
                    queue.poll();
                    queue.offer(arr[i]);
                }
            }
        }
        int[] res = new int[k];
        for (int i = 0; i < k; i ++ ) {
            res[i] = queue.poll();
        }
        return res;
    }
    
    // 方式二：快速排序 + 二分（partition划分的是最终有序数组的下标i，左边是小于nums[i]的无序的数,右边是大于nums[i]的无序的数）
    public int[] getLeastNumbers(int[] arr, int k) {
        if (k == 0 || arr == null || arr.length == 0) return new int[0];        
        return getKnum(arr, k, 0, arr.length - 1);
    }

    int[] getKnum(int[] arr, int k, int l, int r) {
        int i = partition(arr, l, r);
        if (i == k - 1) {
            return Arrays.copyOf(arr, k);
        }
        return i > k - 1 ? getKnum(arr, k, l, i - 1) : getKnum(arr, k, i + 1, r);
    }

    private int partition(int[] arr, int l, int r) {
        int x = arr[l];
        while (l < r) {
            while (l < r && arr[r] > x) r--;
            arr[l] = arr[r];
            while (l < r && arr[l] <= x) l++;
            arr[r] = arr[l]; 
        }
        arr[l] = x;
        return l;
    } 
}
```

## 41 数据流中的中位数

```java
//关键点： 中位数-> 较小数的部分用大顶堆，较大数的部分用小顶堆,去中间数（两数的平均）
如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。
/*
创建大根堆、小根堆，其中：大根堆存放较小的一半元素，小根堆存放较大的一半元素。
添加元素时，若两堆元素个数相等，放入小根堆（使得小根堆个数多 1）；若不等，放入大根堆（使得大小根堆元素个数相等）
取中位数时，若两堆元素个数相等，取两堆顶求平均值；若不等，取小根堆堆顶。

注意： 大根堆多1也行，最后奇数返回大根堆的堆顶即可
*/
class MedianFinder {
    PriorityQueue<Integer> maxHeap;
    PriorityQueue<Integer> minHeap;
    public MedianFinder() {
        // maxHeap = new PriorityQueue<>(Collections.reverseOrder());
        // maxHeap = new PriorityQueue<>((a, b) -> b - a);
        maxHeap = new PriorityQueue(new Comparator<Integer>() {
            public int compare(Integer a, Integer b) {
                // return Integer.compare(b, a);
                return b - a;
            }
        });
        minHeap = new PriorityQueue<>();
    }
    
    public void addNum(int num) {
        if (maxHeap.size() == minHeap.size()) {
            maxHeap.offer(num);
            minHeap.offer(maxHeap.poll()); // 小顶堆加1,大顶堆+1后-1
        } else {
            minHeap.offer(num);
            maxHeap.offer(minHeap.poll());
        }
    }
    
    public double findMedian() {
        if (maxHeap.size() == minHeap.size()) {
            return (maxHeap.peek() + minHeap.peek()) / 2.0;
        } else {
            return minHeap.peek();
        }
    }
}

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder obj = new MedianFinder();
 * obj.addNum(num);
 * double param_2 = obj.findMedian();
 */
```

## 42 连续子数组的最大和

```java
输入一个整型数组，数组里有正数也有负数。数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。
    
class Solution {
    public int maxSubArray(int[] nums) {
        int n = nums.length;
        int[] f = new int[n];
        int res = Integer.MIN_VALUE;
        for (int i = 0; i < n; i ++ ) {
            f[i] = nums[i];
            if (i > 0 && nums[i] + f[i - 1] > f[i]) {
                f[i] = nums[i] + f[i - 1];
            }
            res = Math.max(res, f[i]);
        }
        return res;
    }
}
```

## 43 1~n整数中1出现的次数

```java
/*
输入一个整数 n ，求 1 ～ n 这 n 个整数的十进制表示中 1 出现的次数。

例如，输入 12，1 ～ 12 这些整数中包含 1 的数字有 1、10、11 和 12，1 一共出现了 5 次。
*/
/**
789 hgih=7 low=89
700~789 f(low)
0~6 * 00~99  high * f(base - 1) 这里只统计low的1的个数
hgih=1  -> 189  89+1    low + 1
high!=1 -> 289  1 00~99  100 base
*/
public int countDigitOne(int n) {
    if (n < 1) return 0;
    String s = String.valueOf(n);
    int base = (int) Math.pow(10, s.length() - 1);
    int high = n / base;
    int low = n % base;
    if (high == 1) {
        return countDigitOne(low) + countDigitOne(base - 1) + low + 1;
    } else {
        return countDigitOne(low) + high * countDigitOne(base - 1) + base;
    }
}
```

## 44 数字序列中某一位的数字

```java
/* 数字范围    数量  位数    占多少位
    1-9        9      1       9
    10-99      90     2       180
    100-999    900    3       2700
    1000-9999  9000   4       36000  ...

    例如 2901 = 9 + 180 + 2700 + 12 即一定是4位数,第12位   n = 12;
    数据为 = 1000 + (12 - 1)/ 4  = 1000 + 2 = 1002
    定位1002中的位置 = (n - 1) %  4 = 3    s.charAt(3) = 2;
*/

public int findNthDigit(int n) {
    int digit = 1;  // n所在的数的位数
    long start = 1; // 数据范围的第一个数
    long cnt = 9;  // 占多少位
    while (n > cnt) {
        n -= cnt;
        digit++;
        start *= 10;
        cnt = 9 * digit * start;
    }
    long num = start + (n - 1) / digit;
    return String.valueOf(num).charAt((int)((n - 1) % digit)) - '0';
}
```

## 45 把数组排成最小的数

```java

class Solution {
    public String minNumber(int[] nums) {
        String[] strs = new String[nums.length];
        for (int i = 0; i < nums.length; i ++ ) {
            strs[i] = String.valueOf(nums[i]);
        }

        // Arrays.sort(strs, (a, b) -> (a + b).compareTo(b + a)); // 3 _30 > 30_3 所以30放在3前面
        
        // Arrays.sort(strs, new Comparator<String>() {
        //     @Override
        //     public int compare(String a, String b) {
        //         return (a + b).compareTo(b + a);
        //     }
        // });
        
        qsort(strs, 0, strs.length - 1);

        StringBuilder res = new StringBuilder();
        for (int i = 0; i < strs.length; i ++ ) {
            res.append(strs[i]);
        }
        return res.toString();
    }

    private void qsort(String[] strs, int l, int r) {
        if (l >= r) return;

        int i = l, j = r;
        String x = strs[l + (r - l) / 2];
        while (i < j) {
            while ((strs[i] + x).compareTo(x + strs[i]) < 0) i++;
            while ((strs[j] + x).compareTo(x + strs[j]) > 0) j--;
            if (i <= j) {
                swap(strs, i, j);
                i++;
                j--;
            }
        }
        if (l < j) qsort(strs, l, j);
        if (i < r) qsort(strs, i, r);
    }

    private void swap(String[] strs, int i, int j) {
        String tmp = strs[i];
        strs[i] = strs[j];
        strs[j] = tmp;
    }
}
```

## 46 把数字翻译成字符串

```java
public int translateNum(int num) {
    String s = num + ""; // String.valueOf(num);
    char[] str = s.toCharArray();
    int[] f = new int[str.length + 1];  // 前i个字符翻译的方法数
    f[0] = f[1] = 1;
    for (int i = 2; i <= str.length; i ++ ) {
        f[i] = f[i - 1];
        if (str[i - 2] == '1' || (str[i - 2] == '2' && str[i - 1] < '6')) {
            f[i] += f[i - 2];
        }
    }
    return f[str.length];
}
```



## 47 礼物的最大价值

```java
//在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？ 

public int maxValue(int[][] grid) {
    if (grid == null || grid.length == 0 || grid[0].length == 0) return 0;
    int m = grid.length;
    int n = grid[0].length;
    int[][] f = new int[m][n];
    for (int i = 0; i < m; i ++ ) {
        for (int j = 0; j < n; j ++ ) {
            if (i == 0 && j == 0) f[i][j] = grid[0][0];
            // f[i][j] = Math.max(f[i  - 1][j], f[i][j - 1]) + nums[i][j];
            if (i > 0) f[i][j] = Math.max(f[i][j], f[i - 1][j] + grid[i][j]);
            if (j > 0) f[i][j] = Math.max(f[i][j], f[i][j - 1] + grid[i][j]);
        }
    }
    return f[m - 1][n - 1];
}
```

## 48 最长不含重复字符的子字符串

```java
// 请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。
// 经典滑动窗口问题
public int lengthOfLongestSubstring(String s) {
    char[] str = s.toCharArray();
    Set<Character> set = new HashSet<>();
    int len = 0;
    int l = 0;
    for (int r = 0; r < str.length; r ++ ) {
        while (set.contains(str[r])) {
            set.remove(str[l++]);
        }
        set.add(str[r]);
        len = Math.max(len, r - l + 1);
    }
    return len;
}

public int lengthOfLongestSubstring(String s) {
    char[] str = s.toCharArray();
    Map<Character, Integer> mp = new HashMap<>();
    int i = 0;
    int len = 0;
    for (int j = 0; j < str.length; j ++ ) {
        if (mp.containsKey(str[j])) {
            i = Math.max(i, mp.get(str[j]) + 1);
        }
        mp.put(str[j], j);
        len = Math.max(len, j - i + 1);
    }
    return len;
}
```

## 49 丑数

```java
// 我们把只包含因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。

使用动态规划
public int nthUglyNumber(int n) {
    int[] f = new int[n];
    f[0] = 1;
    int p2 = 0, p3 = 0, p5 = 0;
    for (int i = 1; i < n; i ++ ) {
        int next2 = f[p2] * 2;
        int next3 = f[p3] * 3;
        int next5 = f[p5] * 5;
        f[i] = Math.min(Math.min(next2, next3), next5);
        if (f[i] == next2) p2++;
        if (f[i] == next3) p3++;
        if (f[i] == next5) p5++;
    }
    return f[n - 1];
}
```

## 50 第一个只出现一次的字符

```java
// 在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。

public char firstUniqChar(String s) {
     char[] chars = s.toCharArray();
     if (chars.length == 0) return ' ';
     int[] st = new int[26];
     for (int i = 0; i < chars.length; i ++ ) {
         st[chars[i] - 'a']++;
     }

     for (int i = 0; i < chars.length; i ++ ) {
         if (st[chars[i] - 'a'] == 1) {
             return chars[i];
         }
     }

     return ' ';
 }
```

## 51 数组中的逆序对

```java
在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。

输入: [7,5,6,4]
输出: 5

class Solution {
    // 方法: 归并排序，合并的时候，左边大于右边的数就开始统计cnt
    private int cnt;
    private static int[] tmp = new int[51000];
    public int reversePairs(int[] nums) {
        this.cnt = 0;
        mergeSort(nums, 0, nums.length - 1);
        return cnt;
    }

    public void mergeSort(int[] nums, int l, int r) {
        if (l >= r) return;

        int mid = l + r >> 1;
        mergeSort(nums, l, mid);  // 注意左右数组都有序
        mergeSort(nums, mid + 1, r);
        merge(nums, l, mid, r);
    }
	// 方法一
    public void merge(int[] nums, int l, int mid, int r) {
        // int[] tmp = new int[r - l + 1];
        int idx = 0;
        int i = l;
        int j = mid + 1;
        while (i <= mid && j <= r) {
            if (nums[i] <= nums[j]) {
                tmp[idx++] = nums[i++];
            } else {
                cnt += mid - i + 1;   //  1 2 6 8 9    3 4 7    对于6 > 3 产生【6, 3】【8, 3】【9, 3】逆序对     
                tmp[idx++] = nums[j++];
            }
        }
        while (i <= mid) tmp[idx++] = nums[i++];
        while (j <= r) tmp[idx++] = nums[j++];

        for (int k = 0; k < idx; k ++ ) {
            nums[l + k] = tmp[k];
        }
    }
    // 方法二
    public void merge(int[] nums, int l, int mid, int r) {
        // int[] tmp = new int[r - l + 1];
        int idx = 0;
        int i = l;
        int j = mid + 1;
        while (i <= mid && j <= r) {
            if (nums[i] <= nums[j]) {
                tmp[idx++] = nums[i++];
                cnt += j - (mid + 1);
            } else {   
                tmp[idx++] = nums[j++];
            }
        }
        
        while (i <= mid) {
            tmp[idx++] = nums[i++]; 
            cnt += j - (mid + 1);
        }
        while (j <= r) tmp[idx++] = nums[j++];

        for (int k = 0; k < idx; k ++ ) {
            nums[l + k] = tmp[k];
        }
    }
}
```



## 52 两个链表的第一个公共节点

```java
// 输入两个链表，找出它们的第一个公共节点。
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode curA = headA;
        ListNode curB = headB;
        
        while (curA != curB) {
            curA = curA != null ? curA.next : headB;
            curB = curB != null ? curB.next : headA;
        }

        return curA;
    }
}
```

## 53-i 在排序数组中查找数字I

```java
// 统计一个数字在排序数组中出现的次数。

public int search(int[] nums, int target) {
    if (nums == null || nums.length == 0) return 0;
    int n = nums.length;
    int l = 0, r = n - 1;
    while (l < r) {
        int mid = l + r >> 1;
        if (nums[mid] < target) l = mid + 1;
        else r = mid;
    }
    if (nums[l] != target) return 0;
    int tmp = l;
    l = 0;
    r = n - 1;
    while (l < r) {
        int mid = l + r + 1 >> 1;
        if (nums[mid] > target) r = mid - 1;
        else l = mid;
    }
    return r - tmp + 1;
}
```

## 53-ii 0~n-1中缺失的数字

```java
// 一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。

public int missingNumber(int[] nums) {
    int res = 0;
    int n = nums.length;
    for (int i = 0; i < n; i ++ ) {
        res ^= i ^ nums[i];
    }
    res ^= n;
    return res;
}
```

## 54 二叉搜索树的第k大节点

```java
	// 给定一棵二叉搜索树，请找出其中第 k 大的节点。
	private int res;
    private int cur;
    public int kthLargest(TreeNode root, int k) {
        cur = k;
        res = 0;
        inorder(root);
        return res;
    }
    private void inorder(TreeNode root) {
        if (root == null) return;
        inorder(root.right);
        cur--;
        if (cur == 0) {
            res = root.val;
            return;
        }
        inorder(root.left);
    }
```

## 55-i 二叉树的深度

```java
输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。
    
class Solution {
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
       	// 方式一： 深搜
        // return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
		
        // 方式二: 层次遍历
        int depth = 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            depth++;
            int n = queue.size();
            for (int i = 0; i < n; i ++ ) {
                TreeNode node = queue.poll();
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
        }
        return depth;
    }
}
```

## 55-ii 平衡二叉树

```java
// 输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中任意节点的左右子树的深度相差不超过 1，那么它就是一棵平衡二叉树。

public boolean isBalanced(TreeNode root) {
    if (root == null) return true;
    return isBalanced(root.left) && isBalanced(root.right) && 
        Math.abs(depth(root.left) - depth(root.right)) <= 1;
}   
private int depth(TreeNode root) {
    if (root == null) return 0;
    return Math.max(depth(root.left), depth(root.right)) + 1;
}
```

## 56-i 数组中数字出现的次数

```java
//一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。

public int[] singleNumbers(int[] nums) {
    int res = 0;
    for (int num : nums) {
        res ^= num;
    }
    int flag = 1;
    while ((flag & res) == 0) {
        flag <<= 1;
    }
    int a = 0;
    int b = 0;
    for (int num : nums) {
        if ((num & flag) == 0) a ^= num;
        else b ^= num;
    }
    return new int[]{a, b};
}
```

## 56-ii 数组中数字出现的次数 ii

```java
// 在一个数组 nums 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。

public int singleNumber(int[] nums) {
    // Map<Integer, Integer> mp = new HashMap<>();
    // for (int i = 0; i < nums.length; i ++ ) {
    //     mp.put(nums[i], mp.getOrDefault(nums[i], 0) + 1);
    // }
    // for (Map.Entry<Integer, Integer> enrty : mp.entrySet()) {
    //     if (enrty.getValue() == 1) return enrty.getKey();
    // }
    // return -1;

    /*
        0 ^ x = x
        x ^ x = 0
        x & ~x = 0
        x & ~0 =x
         */
    int a = 0, b = 0;
    for (int num: nums) {
        a = (a ^ num) & ~b;
        b = (b ^ num) & ~a;
    }
    return a;
}
```

## 57-i  和为s的两个数字

```java
// 输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。

class Solution {
    public int[] twoSum(int[] nums, int target) {
        if (nums == null || nums.length == 0) return new int[0];
        int i = 0;
        int j = nums.length - 1;
        while (i < j) {
            if (nums[i] + nums[j] == target) {
                return new int[] {nums[i], nums[j]};
            } else if (nums[i] + nums[j] > target) {
                j--;
            } else {
                i++;
            }
        }
        return new int[0];
    }
}
```

## 57-ii 和为s的连续正数序列

```java
输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。
序列内的数字由小到大排列，不同序列按照首个数字从小到大排列

public int[][] findContinuousSequence(int target) {
    List<int[]> list = new ArrayList<>();
    int l = 1, r = 2;
    while (l < r) {
        int sum = (l + r) * (r - l + 1) / 2;
        if (sum < target) {
            r++;	// 不够大，往右扩大
        } else if (sum > target) {
            l++;  // 以l为边界不存在合理的和
        } else if (sum == target) {
            int[] res = new int[r - l + 1];
            for (int i = l; i <= r; i ++ ) {
                res[i - l] = i;
            }
            list.add(res);
            l++;
        }
    }
    return list.toArray(new int[list.size()][]);
}
```

## 58-i 翻转单词顺序

```java
/*
输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。

输入: "  hello world!  "
输出: "world! hello"
*/
public String reverseWords(String s) {
    String[] strs = s.split("\\s+");
    StringBuffer str = new StringBuffer();
    for (int i = strs.length - 1; i >= 0; i -- ) {
        if (!"".equals(strs[i])) {
            str.append(strs[i]).append(" ");
        }
    }
    if (str.length() == 0) return "";
    return str.deleteCharAt(str.length() - 1).toString(); 
}
```

## 58-ii 左旋转字符串

```java
//字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字 2，该函数将返回左旋转两位得到的结果"cdefgab"。

class Solution {
    public String reverseLeftWords(String s, int k) {
        // 方式一：直接字符拼接
        // return s.substring(k, s.length()) + s.substring(0, k);

        // 方式二：
        // char[] chars = s.toCharArray();
        // StringBuilder str = new StringBuilder();
        // for (int i = k; i < k + chars.length; i ++ ) {
        //     str.append(chars[i % chars.length]);
        // }

        // return str.toString();

        // 方式三
        char[] chars = s.toCharArray();
        
        reverse(chars, 0, k - 1);
        reverse(chars, k, chars.length - 1);
        reverse(chars, 0, chars.length - 1);

        return String.valueOf(chars);
    }

    private void reverse(char[] chars, int l, int r) {
        // for (int i = 0; i < (r - l + 1) / 2; i ++ ) {
        //     char t = chars[l + i];
        //     chars[l + i] = chars[r - i];
        //     chars[r - i] = t;
        // } 

        for (int i = l, j = r; i < j; i++, j-- ) {
            char t = chars[i];
            chars[i] = chars[j];
            chars[j] = t;
        }
    }
} 
```

## 59-i 滑动窗口的最大值

```java
//给定一个数组 nums 和滑动窗口的大小 k，请找出所有滑动窗口里的最大值。
/*
单调队列
q = deque()
for i in range(n):
    # 判断队头是否滑出窗口
    while q and checkout_out(q[0]):
        q.popleft()
    while q and check(q[-1]):
        q.pop()
    q.append(i)
*/
public int[] maxSlidingWindow(int[] nums, int k) {
    if (nums == null || nums.length == 0) return new int[0];

    int idx = 0;
    int n = nums.length;
    int[] res = new int[n - k + 1];

    Deque<Integer> queue = new LinkedList<>();

    for (int i = 0; i < n; i ++ ) {
        while (!queue.isEmpty() && i - queue.peekFirst() + 1 > k) {
            queue.pollFirst();
        }
        while (!queue.isEmpty() && nums[i] > nums[queue.peekLast()]) {
            queue.pollLast();
        }
        queue.offer(i);
        if (i >= k - 1) {
            res[idx++] = nums[queue.peekFirst()];
        }
    }
    return res;
}
```

## 59-ii 队列的最大值

```java
请定义一个队列并实现函数 max_value 得到队列里的最大值，要求函数max_value、push_back 和 pop_front 的均摊时间复杂度都是O(1)。

若队列为空，pop_front 和 max_value 需要返回 -1

class MaxQueue {
    private Deque<Integer> dataQueue;
    private Deque<Integer> maxQueue;
    public MaxQueue() {
        dataQueue = new LinkedList<>();
        maxQueue = new LinkedList<>();
    }
    
    public int max_value() {
        return maxQueue.isEmpty() ? -1 : maxQueue.peekFirst();
    }
    
    public void push_back(int value) {
        while (!maxQueue.isEmpty() && maxQueue.peekLast() < value) {
            maxQueue.pollLast();
        }
        dataQueue.offer(value);
        maxQueue.offer(value);
    }
    
    public int pop_front() {
        if (dataQueue.isEmpty()) return -1;
        int res = dataQueue.pollFirst(); // data必须弹出
        if (res == maxQueue.peekFirst()) maxQueue.pollFirst(); 
        return res; 
    }
}

/**
 * Your MaxQueue object will be instantiated and called as such:
 * MaxQueue obj = new MaxQueue();
 * int param_1 = obj.max_value();
 * obj.push_back(value);
 * int param_3 = obj.pop_front();
 */
```

## 60 n个骰子的点数

```java
//把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。
//你需要用一个浮点数数组返回答案，其中第 i 个元素代表这 n 个骰子所能掷出的点数集合中第 i 小的那个的概率。

public double[] dicesProbability(int n) {
    int[][] f = new int[n + 1][6 * n + 1];

    for (int j = 1; j <= 6; j ++ ) {
        f[1][j] = 1;
    }

    for (int i = 2; i <= n; i ++ ) {
        for (int j = i; j <= 6 * i; j ++ ) {
            for (int k = 1; k <= 6 && j > k; k ++ ) {
                f[i][j] += f[i - 1][j - k];
            }
        }
    }

    double[] res = new double[5 * n + 1];
    double all = Math.pow(6, n);
    for (int i = 0; i < 5 * n + 1; i ++ ) {
        res[i] = f[n][n + i] * 1.0 / all;
    }

    return res;
}
```

## 61 扑克牌中的顺子

```java
//从若干副扑克牌中随机抽 5 张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。

public boolean isStraight(int[] nums) {
    if (nums == null || nums.length == 0) return false;

    Arrays.sort(nums);

    int joker = 0;
    for (int i = 0; i < 4; i ++ ) {
        if (nums[i] == 0) {
            joker++;
        } else if (nums[i] == nums[i + 1]) {
            return false;
        }
    }

    return nums[4] - nums[joker] <= 4;
}

// 方法二
int minValue = Integer.MAX_VALUE;

boolean[] t = new boolean[14];
for (int i = 0; i < nums.length; i ++ ) {
    if (nums[i] == 0) continue;

    if (t[nums[i]] == true) return false;

    t[nums[i]] = true;

    maxValue = Math.max(maxValue, nums[i]);
    minValue = Math.min(minValue, nums[i]);
}
return maxValue - minValue <= 4;
```

## 62 圆圈中最后剩下的数字

```java
// 0,1,,n-1 这 n 个数字排成一个圆圈，从数字 0 开始，每次从这个圆圈里删除第 m 个数字。求出这个圆圈里剩下的最后一个数字。

//例如，0、1、2、3、4 这 5 个数字组成一个圆圈，从数字 0 开始每次删除第 3 个数字，则删除的前 4 个数字依次是 2、0、4、1，因此最后剩下的数字是 3。

public int lastRemaining(int n, int m) {
    // Queue<Integer> q = new LinkedList<>();
    // for (int i = 0; i < n; i ++ ) {
    //     q.offer(i);
    // }
    // int res = -1;
    // while (q.size() > 1) {
    //     for (int i = 0; i < m - 1; i ++ ) {
    //         q.offer(q.poll());
    //     }
    //     q.poll();
    // }
    // return q.poll();

    // // 方法二：
    List<Integer> list = new ArrayList<>();
    for (int i = 0; i < n; i ++ ) {
        list.add(i);
    }
    int idx = 0;
    while (list.size() > 1) {
        idx = (idx + m - 1) % list.size();
        list.remove(idx);
    }
    return list.get(0);

    // // 方法三：
    // // int res = 0;
    // // for (int i = 1; i <= n; i ++ ) {
    // //     res = (res + m) % i;
    // // }

    // // return res;
}
```

## 63 股票的最大利润

```java
// 假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？

class Solution {
    public int maxProfit(int[] prices) {
        if (prices == null | prices.length == 0) return 0;
        // 注意是买卖一次
        int res = 0;
        int min = prices[0];
        for (int i = 1; i < prices.length; i ++ ) {
            min = Math.min(min, prices[i]);
            res = Math.max(res, prices[i] - min);
        }
        return res;
    }
}
```

## 64 求1+2+…+n

```java
// 求1+2+…+n
// 要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）
public int sumNums(int n) {
    boolean flag = n > 0 && (n += sumNums(n - 1)) > 0;
    return n;
}
```

## 65 不用加减乘除做加法

```java
写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。
// 迭代方法
public int add(int a, int b) {
    // (a ^ b) ^ ((a & b) << 1)
    while (b != 0) {
        int sum = a ^ b;
        b = (a & b) << 1;
        a = sum;
    }
    return a;
    
    // 递归方法
    if (b == 0) return a;
    return add((a ^ b), ((a & b) << 1));
}
```

## 66 构建乘积数组

```java
给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，其中 B 中的元素 B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法。
    
public int[] constructArr(int[] a) {
    int n = a.length;
    int[] res = new int[n];
    for (int i = 0, left = 1; i < n; i ++ ) {
        res[i] = left;
        left *= a[i];
    }
    for (int i = n - 1, right = 1; i >= 0; i -- ) {
        res[i] *= right;
        right *= a[i];
    }
    return res;
}
```

## 67 把字符串转换成整数

```java
写一个函数 StrToInt，实现把字符串转换成整数这个功能。不能使用 atoi 或者其他类似的库函数。
假设我们的环境只能存储 32 位大小的有符号整数，那么其数值范围为 [−231,  231 − 1]。如果数值超过这个范围，请返回  INT_MAX (231 − 1) 或 INT_MIN (−231) 。
在任何情况下，若函数不能进行有效的转换时，请返回 0。
    
public int strToInt(String str) {
    if (str == null || str.trim().length() == 0) return 0;
    char[] s = str.toCharArray();
    int i = 0;
    while (i < s.length && s[i] == ' ') i++;
    int sign = 1;
    if (s[i] == '-') {
        sign = -1;
        i++;
    } else if (s[i] == '+') {
        i++;
    }

    int res = 0;
    boolean hasNum = false;
    for (; i < s.length; i ++ ) {
        int num = s[i] - '0';
        if ((num < 0 || num > 9) && hasNum == false) return 0;
        if ((num < 0 || num > 9) && hasNum == true) break;
        hasNum = true; 
        if (res > (Integer.MAX_VALUE - num) / 10) {
            return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
        }
        res = res * 10 + num;
    }
    return sign * res;
}
```

## 68-i 二叉搜索树的最近公共祖先

```java
/*
给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。
百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
例如，给定如下二叉搜索树:  root = [6,2,8,0,4,7,9,null,null,3,5]
*/

class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (p == q) return p;
        if (root == null) return null;

        // if (root.val < p.val && root.val < q.val) {
        //     return lowestCommonAncestor(root.right, p, q);
        // }

        // if (root.val > p.val && root.val > q.val) {
        //     return lowestCommonAncestor(root.left, p, q);
        // }

        // return root;

        while (root != null) {
            if (root.val < p.val && root.val < q.val) {
                root = root.right;
            } else if (root.val > p.val && root.val > q.val) {
                root = root.left;
            } else {
                break;
            }
        }

        return root;
    }
}
```

## 68-ii 二叉树的最近公共祖先

```java
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

例如，给定如下二叉树:  root = [3,5,1,6,2,0,8,null,null,7,4]

class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return root;
        if (root == p || root == q) return root;

        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);

        if (left == null) return right;
        if (right == null) return left;

        return root;
    }
    	if (root == null) return null;
        if (root == p || root == q) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left != null && right != null) return root;
        if (left != null && right == null) return left;
        if (right != null && left == null) return right;
        return null;
}
```

# Carl习题集

## 背包问题总结

![](.\pic\leetcode\背包问题递推公式.png)

![](.\pic\leetcode\背包问题遍历顺序.png)

```c++
背包问题总结 (一般OJ的数组从0开始， 第一层for循环可以修改为0开始，下面不变)
1、存在型背包： f[j] = f[j] || f[j - v[i]]
2、价值型背包： f[j] = max(f[j], f[j - v[i]] + w[i]) 
				f[j] = max(f[j], f[j - k * v[i]] + k * w[i]) 
3、方案数型背包： f[j] += f[j - v[i]]
4、概率型背包   f[j] = min(f[j], f[j - v[i]] * (1.0 - probability[i])) 
				 //f[j] 表示前i个学校，本金j不中的概率（全部不中） 

背包一（存在型01背包）f[0] = 1  f[1 ~ m] = 0
		for(int i = 1; i <= n; i ++ ) {
            for(int j = m; j >= A[i - 1]; j -- ) {
                f[j] = f[j] || f[j - A[i - 1]];
            }
        }

背包二 (价值型01背包) f[0] = 0   f[1 ~ m] = 0 体积小于等于i的赋值， 不需遍历 f[m] 
					  f[0] = 0   f[1 ~ m] = INF(求最小)/-INF(求最大) 体积恰好为i，要遍历 
						
		for(int i = 1; i <= n; i ++ ) {
			for(int j = m; j >= v[i - 1]; j -- ) {
				f[j] = max(f[j], f[j - v[i - 1]] + w[i - 1]); 
			} 
		}
背包三 （价值型完全背包）f[0] = 0   f[1 ~ m] = 0 体积小于等于i的赋值， 不需遍历 f[m] 
					     f[0] = 0   f[1 ~ m] = INF(求最小)/-INF(求最大) 体积恰好为i，要遍历 
		for(int i = 1; i <= n; i ++ ) {
            for(int j = v[i - 1]; j <= m; j ++ ) {
                f[j] = max(f[j], f[j - v[i - 1]] + w[i - 1]);
            }
        }
        
        (价值型多重背包) f[0] = 0   f[1 ~ m] = 0 体积小于等于i的赋值， 不需遍历 f[m] 
					     f[0] = 0   f[1 ~ m] = INF(求最小)/-INF(求最大) 体积恰好为i，要遍历 
		for(int i = 1; i <= n; i ++ ) {
			for(int j = m; j <= v[i]; j ++ ) {
				for(int k = 0; k <= s[i] && k * v[i] <= j; k ++ ) {
					f[j] = max(f[j], f[j - k * v[i]] + k * w[i]);
				}
			}
		} 

背包四 （方案型完全背包：无限次 + 无顺序）f[0] = 1   f[1 ~ m] = 0
		for(int i = 1; i <= n; i ++ ) {
            for(int j = A[i - 1]; j <= m; j ++ ) {
                f[j] += f[j - A[i - 1]];
            }
        }
/*
在求装满背包有几种方案的时候，认清遍历顺序是非常关键的。

如果求组合数就是外层for循环遍历物品，内层for遍历背包。
如果求排列数就是外层for遍历背包，内层for循环遍历物品。
*/
 
背包五 （方案型01背包：只有一次）f[0] = 1    f[1 ~ m] = 0
		for(int i = 1; i <= n; i ++ ) {
            for(int j = m; j >= A[i - 1]; j -- ) {
                f[j] += f[j - A[i - 1]];
            }
        }

背包六 （方案型完全背包：无限次 + 有顺序）	 f[0] = 1   f[1 ~ m] = 0
		for(int j = 1; j <= m; j ++ ) {  //先遍历体积
            for(int i = 1; i <= n; i ++ ) { //再遍历物品 
                if(j >= A[i - 1]) {
                    f[j] += f[j - A[i - 1]];
                }
            }
        }

背包七	（多重背包:珍爱生活）
		cin >> v[i] >> w[i] >> s[i];
		for(int i = 0; i < n; i ++ ) {
            for(int j = m; j >= v[i]; j -- ) {
                for(int k = 0; k <= s[i] && k * v[i] <= j; k ++ ) {
                    f[j] = max(f[j], f[j - k * v[i]] + k * w[i]);
                }
            }
        } 
	

背包八 统计方案数 （f[i] 表示能否组成i, 另外开数组统计硬币数目） 
		（给一些不同价值和数量的硬币。找出这些硬币可以组合在1 ~ n范围内的值的数量） 
		for(int i = 0; i < n; i ++ ) {
            vector<int> cnt(m + 1, 0);
            for(int j = v[i]; j <= m; j ++ ) {
                if(!f[j] && f[j - v[i]] && cnt[j - v[i]] < s[i]) {
                    f[j] = 1;
                    cnt[j] += cnt[j - v[i]] + 1;
                    res++;
                }
            }
        }
        return res;
      
背包九  （概率背包）
	   vector<double> f(m + 1, 1.0);  //前i个学校， 有n元时， 没有offer的最低概率
       
       for(int i = 1; i <= n; i ++ ) { //0-1背包问题
           for(int j = m; j >= v[i]; j -- ) {
               f[j] = min(f[j], f[j - v[i]] * (1.0 - probability[i]));
           }
       }
       
       return 1 - f[m];        

背包十  （完全背包变形之给小费：总 - max f[i]） 
		int v[3] = {a, b, c}; 
		for(int i = 0; i < 3; i ++ ) {
            for(int j = v[i]; j <= m; j ++ ) {
                f[j] = max(f[j], f[j - v[i]] + v[i]);
            }
        }
        return m - f[m];

背包变形
1】  0-1背包问题的路径打印 （0-1背包的路径打印 （字典序最小如： 123 < 31）） 
	for(int i = n; i >= 1; i -- ) {
        for(int j = 0; j <= m; j ++ ) {
            f[i][j] = f[i + 1][j];
            if(j >= v[i]) {
                f[i][j] = max(f[i][j], f[i + 1][j - v[i]] + w[i]);
            }
        }
    }
    
    int j = m;
    for(int i = 1; i <= n; i ++ ) {
        if(j >= v[i] && f[i][j] == f[i + 1][j - v[i]] + w[i]) {
            cout << i << " ";
            j -= v[i];
        }
    } 
2】  0-1背包最大价值的方案数  
const int N = 1010, INF = 1e9, mod = 1e9 + 7;
int f[N], g[N]; //f[i]表示容量恰好为i时的价值， g[i]表示该容量的方案数

    cin >> n >> m;
    
    for(int i = 1; i <= n; i ++ ) cin >> v[i] >> w[i];
    
    g[0] = 1; //方案
    f[0] = 0; //价值
    for(int i = 1; i <= m; i ++ ) f[i] = -INF;   //恰好为, 求最大值 
    
    for(int i = 1; i <= n; i ++ ) {
        for(int j = m; j >= v[i]; j -- ) {
            // f[j] = max(f[j], f[j - v[i]] + w[i]); 相当于拆开统计
            int maxv = max(f[j], f[j - v[i]] + w[i]);
            int cnt = 0;
            if(maxv == f[j]) cnt += g[j];
            if(maxv == f[j - v[i]] + w[i]) cnt += g[j - v[i]];
            g[j] = cnt % mod;
            f[j] = maxv;
        }
    }
    
    int res = 0;
    for(int i = 0; i <= m; i ++ ) res = max(res, f[i]);
    // cout << res << endl;
    
    int cnt = 0;
    for(int i = 0; i <= m; i ++ ) {
        if(res == f[i]) {
            cnt = (cnt + g[i]) % mod;
        }
    }
    cout << cnt << endl;


 
============================================================= 
1）背包一：（求最大重量，01背包存在性问题） 
在n个物品中挑选若干物品装入背包，最多能装多满？
假设背包的大小为m，每个物品的大小为A[i] 

样例 1:
	输入:  [3,4,8,5], backpack size=10
	输出:  9

样例 2:
	输入:  [2,3,5,7], backpack size=12
	输出:  12
	
	
class Solution {
public:
    /**
     * @param m: An integer m denotes the size of a backpack
     * @param A: Given n items with size A[i]
     * @return: The maximum size
     */
    int backPack(int m, vector<int> &A) {
        // int n = A.size();
        
        // if(n == 0) {
        //     return 0;
        // }
        
        // int f[n + 1][m + 1];
        
        // for(int i = 1; i <= m; i ++ ) {
        //     f[0][i] = 0;
        // }
        
        // f[0][0] = 1;
        
        // for(int i = 1; i <= n; i ++ ) {
        //     for(int j = 0; j <= m; j ++ ) {
        //         f[i][j] = f[i - 1][j];
        //         if(j >= A[i - 1]) {
        //             f[i][j] = f[i][j] || f[i - 1][j - A[i - 1]];
        //         }
        //     }
        // }
        
        // for(int i = m; i >= 0; i -- ) {
        //     if(f[n][i]) {
        //         return i;
        //     }
        // }
        
        // return 0;
        int n = A.size();
        if(n == 0) return 0;
        
        bool f[m + 1]; //能否装体积恰好是i
        
        for(int i = 1; i <= m; i ++ ) f[i] = 0; //恰好装满体积i的初始条件
        
        f[0] = 1;
        
        for(int i = 1; i <= n; i ++ ) {
            for(int j = m; j >= A[i - 1]; j -- ) {
                f[j] = f[j] || f[j - A[i - 1]];
            }
        }
        
        for(int i = m; i >= 0; i -- ) {
            if(f[i] == 1) {
                return i;
            }
        }
        
        return 0;
    }
};


2）背包二：（求价值， 01背包） 
有 n 个物品和一个大小为 m 的背包. 
给定数组 A 表示每个物品的大小和数组 V 表示每个物品的价值.
问最多能装入背包的总价值是多大? 每个物品只能取一次

样例 1:

输入: m = 10, A = [2, 3, 5, 7], V = [1, 5, 2, 4]
输出: 9
解释: 装入 A[1] 和 A[3] 可以得到最大价值, V[1] + V[3] = 9 
样例 2:

输入: m = 10, A = [2, 3, 8], V = [2, 5, 8]
输出: 10
解释: 装入 A[0] 和 A[2] 可以得到最大价值, V[0] + V[2] = 10

class Solution {
public:
    /**
     * @param m: An integer m denotes the size of a backpack
     * @param A: Given n items with size A[i]
     * @param V: Given n items with value V[i]
     * @return: The maximum value
     */
    int backPackII(int m, vector<int> &v, vector<int> &w) {
        if(v.empty() || w.empty()) return 0;
        
        int n = v.size();
        
        int f[m + 1];  //体积为 i 的最大价值（可以不装满）
        
        for(int i = 0; i <= m; i ++ ) {
            f[i] = 0;
        }
        
        for(int i = 1; i <= n; i ++ ) {
            for(int j = m; j >= v[i - 1]; j -- ) {
                f[j] = max(f[j], f[j - v[i - 1]] + w[i - 1]);
            }
        }
        
        return f[m];
        // if(v.empty() || w.empty()) return 0;
        
        // int n = v.size();
        
        // int f[n + 1][m + 1];  //f[i][j] 表示前 i 物品,组成 体积为 j的最大价值 
        
        // for(int i = 1; i <= m; i ++ ) f[0][i] = -1;
        
        // f[0][0] = 0;
        // for(int i = 1; i <= n; i ++ ) {
        //     for(int j = 0; j <= m; j ++ ) {
        //         f[i][j] = f[i - 1][j];
        //         if(j >= v[i - 1] && f[i - 1][j - v[i - 1]] != -1) {
        //             f[i][j] = max(f[i][j], f[i - 1][j - v[i - 1]] + w[i - 1]);
        //         }
        //     }
        // }
        
        // int res = 0;
        // for(int i = 0; i <= m; i ++ ) {
        //     res = max(res, f[n][i]);
        // }
        
        // return res;
        
        // int n = v.size();
        
        // if(n == 0) {
        //     return 0;
        // }
        
        // int f[n + 1][m + 1];
        // int g[n + 1][m + 1];
        
        // f[0][0] = 0;
        // for(int i = 1; i <= m ;i ++ ) {
        //     f[0][i] = 0;
        // }
        
        // for(int i = 1; i <= n; i ++ ) {
        //     for(int j = 0; j <= m; j ++ ) {
        //         f[i][j] = f[i - 1][j];
        //         g[i][j] = 0;
        //         if(j >= v[i - 1]) {
        //             f[i][j] = max(f[i][j], f[i - 1][j - v[i - 1]] + w[i - 1]);
        //             if(f[i][j] == f[i - 1][j - v[i - 1]] + w[i - 1]) {
        //                 g[i][j] = 1;
        //             }
        //         }
        //     }
        // }
        
        // int j = m;
        // int path[n];
        // for(int i = n; i >= 1; i -- ) {
        //     if(g[i][j] == 1) {
        //         path[i - 1] = true;
        //         j -= v[i - 1];
        //     } else {
        //         path[i - 1] = false;
        //     }
        // }
        
        // for(int i = 0; i < n; i ++ ) {
        //     // if(path[i] == 1) {
        //     //     printf("%d %d\n",v[i], w[i]);
        //     // }
        //   if(path[i] == 1) {
        //       cout << "item " << i << " " << v[i] << " " << w[i] << endl;
        //   }
        // } 

        // return f[n][m];
        // int n = v.size();
        // if(n == 0) {
        //     return 0;
        // }
        
        // int f[m + 1];
        
        // memset(f, 0, sizeof(f));
        
        // for(int i = 1; i <= n; i ++ ) {
        //     for(int j = m; j >= v[i - 1]; j -- ) {
        //         f[j] = max(f[j], f[j - v[i - 1]] + w[i - 1]);
        //     }
        // }
        
        // return f[m];
    }
};

3）背包三：（完全背包问题） 
给定 n 种物品, 每种物品都有无限个. 第 i 个物品的体积为 A[i], 价值为 V[i].

再给定一个容量为 m 的背包. 问可以装入背包的最大价值是多少?

样例 1:

输入: A = [2, 3, 5, 7], V = [1, 5, 2, 4], m = 10
输出: 15
解释: 装入三个物品 1 (A[1] = 3, V[1] = 5), 总价值 15.
样例 2:

输入: A = [1, 2, 3], V = [1, 2, 3], m = 5
输出: 5
解释: 策略不唯一. 比如, 装入五个物品 0 (A[0] = 1, V[0] = 1). 

class Solution {
public:
    /**
     * @param A: an integer array
     * @param V: an integer array
     * @param m: An integer
     * @return: an array
     */
    int backPackIII(vector<int> &v, vector<int> &w, int m) {
        if(v.empty() || w.empty()) return 0;
        
        int n = v.size();
        
        int f[m + 1];   //f[i] 表示背包体积是 i 的最大价值（背包体积可以不装满）
        
        for(int i = 0; i <= m; i ++ ) {
            f[i] = 0;
        }
        
        for(int i = 1; i <= n; i ++ ) {
            for(int j = v[i - 1]; j <= m; j ++ ) {
                f[j] = max(f[j], f[j - v[i - 1]] + w[i - 1]);
            }
        }
        
        return f[m];
        
        // int n = v.size();
        
        // if(n == 0) {
        //     return 0;
        // }
        
        // int f[n + 1][m + 1];
        
        // memset(f, 0, sizeof(f)); 
        
        // for(int i = 1; i <= n; i ++ ) {
        //     for(int j = 0; j <= m; j ++ ) {
        //         for(int k = 0; k * v[i - 1] <= j; k ++ ) {
        //             f[i][j] = max(f[i][j], f[i - 1][j - v[i - 1] * k] + w[i - 1] * k);
        //         }
        //     }
        // }
        
        // return f[n][m];
    }
};

背包三-2 多重背包问题
	for(int i = 1; i <= n; i ++ ) {
		for(int j = m; j >= v[i]; j -- ) {
			for(int k = 0; k * v[i] <= j && k <= s[i]; k ++ ) {
				f[j] = max(f[j], f[j - v[i] * k] + w[i] * k);
			}
		}
	}
	cout << f[m]; 

4）背包四 （无限次 + 不考虑顺序） 
给出 n 个物品, 以及一个数组, nums[i]代表第i个物品的大小, 
保证大小均为正数并且没有重复, 正整数 target 表示背包的大小, 
找到能填满背包的方案数。
每一个物品可以使用无数次

样例1

输入: nums = [2,3,6,7] 和 target = 7
输出: 2
解释:
方案有: 
[7]
[2, 2, 3]
样例2

输入: nums = [2,3,4,5] 和 target = 7
输出: 3
解释:
方案有: 
[2, 5]
[3, 4]
[2, 2, 3]

class Solution {
public:
    /**
     * @param nums: an integer array and all positive numbers, no duplicates
     * @param target: An integer
     * @return: An integer
     */
    int backPackIV(vector<int> &A, int m) {
        
        // vector<int> dp(target + 1);
        // dp[0] = 1;
        // for (auto a : nums) {
        //     for (int i = a; i <= target; ++i) {
        //         dp[i] += dp[i - a];
        //     }
        // }
        // return dp.back();
        
        if(A.empty()) return 0;
        
        int n = A.size();
        
        int f[m + 1]; // f[i] 表示恰好体积为 i 的方案数
        
        for(int i = 1; i <= m; i ++ ) {
            f[i] = 0;
        }
        
        f[0] = 1;
        
        for(int i = 1; i <= n; i ++ ) {
            for(int j = A[i - 1]; j <= m; j ++ ) {
                f[j] += f[j - A[i - 1]];
            }
        }
    
        return f[m];
       
     }
     
};


5）背包五 （只能用一次） 
给出 n 个物品, 以及一个数组, nums[i] 代表第i个物品的大小, 保证大小均为正数,
 正整数 target 表示背包的大小, 找到能填满背包的方案数。
每一个物品只能使用一次

样例
给出候选物品集合 [1,2,3,3,7] 以及 target 7

结果的集合为:
[7]
[1,3,3]
返回 2

class Solution {
public:
    /**
     * @param nums: an integer array and all positive numbers
     * @param target: An integer
     * @return: An integer
     */
    int backPackV(vector<int> &A, int m) {
        // int n = A.size();
        
        // if(n == 0) {
        //     return 0;
        // }
        
        // int f[n + 1][m + 1]; //前i 个物品拼出 重量位j的方式数目（不是存在性）
        
        // f[0][0] = 1;
        // //前0个物品拼不出任何重量[1 ~ m] 
        // for(int i = 1; i <= m; i ++ ) {
        //     f[0][i] = 0; 
        // }
        
        // for(int i = 1; i <= n; i ++ ) {
        //     for(int j = 0; j <= m; j ++ ) {
        //         f[i][j] = f[i - 1][j];
        //         if(j >= A[i - 1]) {
        //             // f[i][j] = max(f[i][j], f[i - 1][j - v[i]] + w[i]);   //0~1背包问题
        //             f[i][j] = f[i][j] + f[i - 1][j - A[i - 1]];
        //         }
        //     }
        // }
        
        // return f[n][m];
        /*
        vector<int> dp(target + 1);
        dp[0] = 1;
        for (auto a : nums) {
            for (int i = target; i >= a; --i) {
                dp[i] += dp[i - a];
            }
        }
        return dp.back();
        */
        // int n = A.size();
        
        // if(n == 0) {
        //     return 0;
        // }
        
        // int f[m + 1];
        
        // f[0] = 1;
        
        // for(int i = 1; i <= m; i ++ ) {
        //     f[i] = 0;
        // }
        
        // for(int i = 1; i <= n; i ++ ) {
        //     for(int j = m; j >= A[i - 1]; j -- ) {
        //         f[j] += f[j - A[i - 1]];
        //     }
        // }
        
        // return f[m];
        if(A.empty()) return 0;
        
        int n = A.size();
        
        int f[m + 1]; // 01背包， f[i] 表示体积恰好为i的方案数目
        
        for(int i = 1; i <= m; i ++ ) {
            f[i] = 0;
        }
        
        f[0] = 1;
        
        for(int i = 1; i <= n; i ++ ) {
            for(int j = m; j >= A[i - 1]; j -- ) {
                f[j] += f[j - A[i - 1]];
            }
        }
        
        return f[m];
    }
};

6） 背包六 （无限次 + 考虑顺序） 
给出一个都是正整数的数组 nums，其中没有重复的数。
从中找出所有的和为 target 的组合个数。
一个数可以在组合中出现多次。
数的顺序不同则会被认为是不同的组合。

样例1

输入: nums = [1, 2, 4] 和 target = 4
输出: 6
解释:
可能的所有组合有：
[1, 1, 1, 1]
[1, 1, 2]
[1, 2, 1]
[2, 1, 1]
[2, 2]
[4]
样例2

输入: nums = [1, 2] 和 target = 4
输出: 5
解释:
可能的所有组合有：
[1, 1, 1, 1]
[1, 1, 2]
[1, 2, 1]
[2, 1, 1]
[2, 2]

class Solution {
public:
    /**
     * @param nums: an integer array and all positive numbers, no duplicates
     * @param target: An integer
     * @return: An integer
     */
    int backPackVI(vector<int> &A, int m) {
        /*
        vector<int> dp(target + 1);
        dp[0] = 1;
        for (int i = 1; i <= target; ++i) {
            for (auto a : nums)
            if (i >= a) {
                dp[i] += dp[i - a];
            }
        }
        return dp.back();
        */
        if(A.empty() || A.size() == 0 || m == 0 ) return 0;
        
        int f[m + 1];  //体积为恰好为 i 的，的无限次的，有顺序的方案数
        
        int n = A.size();
        
        for(int i = 1; i <= m ; i ++ ) {
            f[i] = 0;
        }
        
        f[0] = 1;
        
        for(int j = 1; j <= m; j ++ ) {  //先遍历体积
            for(int i = 1; i <= n; i ++ ) {
                if(j >= A[i - 1]) {
                    f[j] += f[j - A[i - 1]];
                }
            }
        }
        
        return f[m];
    }
};

7）背包 VII （多重背包） 

假设你身上有 n 元，超市里有多种大米可以选择，每种大米都是袋装的，必须整袋购买，
给出每种大米的价格，重量以及数量，求最多能买多少公斤的大米

样例 1:

输入:  n = 8, prices = [3,2], weights = [300,160], amounts = [1,6]
输出:  640	
解释:  全买价格为2的米。
样例 2:

输入:  n = 8, prices  = [2,4], weight = [100,100], amounts = [4,2 ]
输出:  400	
解释:  全买价格为2的米	

class Solution {
public:
    /**
     * @param n: the money of you
     * @param prices: the price of rice[i]
     * @param weight: the weight of rice[i]
     * @param amounts: the amount of rice[i]
     * @return: the maximum weight
     */
    int backPackVII(int m, vector<int> &v, vector<int> &w, vector<int> &s) {
        if(w.empty() || v.empty() || s.empty()|| m == 0) return 0;
        
        int n = v.size();
        
        vector<int> f(m + 1);
        
        f[0] = 0;
        for(int i = 1; i <= m; i ++ ) f[i] = 0;
        
        for(int i = 0; i < n; i ++ ) {
            for(int j = m; j >= v[i]; j -- ) {
                for(int k = 0; k <= s[i] && k * v[i] <= j; k ++ ) {
                    f[j] = max(f[j], f[j - k * v[i]] + k * w[i]);
                }
            }
        }
        return f[m];
    }
}; 

8）背包 VIII
给一些不同价值和数量的硬币。找出这些硬币可以组合在1 ~ n范围内的值的数量

样例 1:
	输入:  
		n = 5
		value = [1,4]
		amount = [2,1]
	输出:  4
	
	解释:
	可以组合出1，2，4，5

样例 2:
	输入: 
		n = 10
		value = [1,2,4]
		amount = [2,1,1]
	输出:  8
	
	解释:
	可以组合出1-8所有数字。

class Solution {
public:
    /**
     * @param n: the value from 1 - n
     * @param value: the value of coins
     * @param amount: the number of coins
     * @return: how many different value
     */
    int backPackVIII(int m, vector<int> &v, vector<int> &s) {
        int n = v.size();
        
        vector<int> f(m + 1, 0); // 是否存在方案，组成m
        
        f[0] = 1;
        for(int i = 1; i <= m; i ++ ) f[i] = 0;
        
        int res = 0;
        
        for(int i = 0; i < n; i ++ ) {
            vector<int> cnt(m + 1, 0);
            for(int j = v[i]; j <= m; j ++ ) {
                if(!f[j] && f[j - v[i]] && cnt[j - v[i]] < s[i]) {
                    f[j] = 1;
                    cnt[j] += cnt[j - v[i]] + 1;
                    res++;
                }
            }
        }
        return res;
    }
};


9）背包 IX（概率）

你总共有n元，商人总共有三种商品，它们的价格分别是150元,250元,350元，
三种商品的数量可以认为是无限多的，购买完商品以后需要将剩下的钱给商人作为小费，
求最少需要给商人多少小费

您在真实的面试中是否遇到过这个题？  
样例
样例 1:

输入:  n = 900
输出:  0
样例 2:

输入:  n = 800
输出:  0
class Solution {
public:
    /**
     * @param n: Your money
     * @param prices: Cost of each university application
     * @param probability: Probability of getting the University's offer
     * @return: the  highest probability
     */
    double backpackIX(int m, vector<int> &v, vector<double> &probability) {
       //至少获得一份offer的最高可能 = 1 - 全部都获取不了的最低可能
       int n = v.size();
       
       vector<double> f(m + 1, 1.0);  //前i个学校， 有n元时， 没有offer的最低概率
       
       for(int i = 1; i <= n; i ++ ) { //0-1背包问题
           for(int j = m; j >= v[i - 1]; j -- ) {
               f[j] = min(f[j], f[j - v[i - 1]] * (1.0 - probability[i - 1]));
           }
       }
       
       return 1 - f[m];
    }
}; 

10）背包 X 
你总共有n元，商人总共有三种商品，它们的价格分别是150元,250元,350元，
三种商品的数量可以认为是无限多的，购买完商品以后需要将剩下的钱给商人作为小费，
求最少需要给商人多少小费

样例
样例 1:

输入:  n = 900
输出:  0
样例 2:

输入:  n = 800
输出:  0


class Solution {
public:
    /**
     * @param n: the money you have
     * @return: the minimum money you have to give
     */
    int backPackX(int m) {
        int v[3] = {150, 250, 350};
        //最少给小费，即求最大消费 (m - f[m])
        
        vector<int> f(m + 1);
        
        f[0] = 0;
        for(int i = 1; i <= m; i ++ ) f[i] = 0;
        
        for(int i = 0; i < 3; i ++ ) {
            for(int j = v[i]; j <= m; j ++ ) {
                f[j] = max(f[j], f[j - v[i]] + v[i]);
            }
        }
        return m - f[m];
    }
};

11） 0-1背包问题的路径打印 （0-1背包的路径打印 （字典序最小如： 123 < 31）） 
	for(int i = n; i >= 1; i -- ) {
        for(int j = 0; j <= m; j ++ ) {
            f[i][j] = f[i + 1][j];
            if(j >= v[i]) {
                f[i][j] = max(f[i][j], f[i + 1][j - v[i]] + w[i]);
            }
        }
    }
    
    int j = m;
    for(int i = 1; i <= n; i ++ ) {
        if(j >= v[i] && f[i][j] == f[i + 1][j - v[i]] + w[i]) {
            cout << i << " ";
            j -= v[i];
        }
    } 

12)  0-1背包最大价值的方案数 
#include <bits/stdc++.h>

using namespace std;

const int N = 1010, INF = 1e9, mod = 1e9 + 7;
int v[N], w[N];
int f[N], g[N]; //f[i]表示容量恰好为i时的价值， g[i]表示该容量的方案数
int n, m;

int main() {
    
    cin >> n >> m;
    
    for(int i = 1; i <= n; i ++ ) cin >> v[i] >> w[i];
    
    g[0] = 1; //方案
    f[0] = 0; //价值
    for(int i = 1; i <= m; i ++ ) f[i] = -INF;   //恰好为, 求最大值 
    
    for(int i = 1; i <= n; i ++ ) {
        for(int j = m; j >= v[i]; j -- ) {
            // f[j] = max(f[j], f[j - v[i]] + w[i]); 相当于拆开统计
            int maxv = max(f[j], f[j - v[i]] + w[i]);
            int cnt = 0;
            if(maxv == f[j]) cnt += g[j];
            if(maxv == f[j - v[i]] + w[i]) cnt += g[j - v[i]];
            g[j] = cnt % mod;
            f[j] = maxv;
        }
    }
    
    int res = 0;
    for(int i = 0; i <= m; i ++ ) res = max(res, f[i]);
    // cout << res << endl;
    
    int cnt = 0;
    for(int i = 0; i <= m; i ++ ) {
        if(res == f[i]) {
            cnt = (cnt + g[i]) % mod;
        }
    }
    cout << cnt << endl;
     
    return 0;
}
```



## 股票问题

```java
（一）买卖股票最佳时机 121   k = 1
/*
你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
（单次买卖）低价购入，高价卖出
*/
public int maxProfit(int[] prices) {
    // 低价购入，高价卖出（单次买卖）
    int res = 0; 
    int min = Integer.MAX_VALUE;
    for (int i = 0; i < prices.length; i ++ ) {
        //min = Math.min(min, prices[i]);
        //res = Math.max(res, prices[i] - min);
    	if (prices[i] < min) {
            min = prices[i];
        } else if (prices[i] - min > res) {
            res = prices[i] - min;
        }
    }
    return res;
}   

(二）买卖股票的最佳时机 II 122  k = ∞
/*
设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

（多次买卖）有利就可 res = max(res, nums[i] - nums[i - 1])
*/
public int maxProfit(int[] prices) {
    int res = 0;
    for (int i = 0; i < prices.length; i ++ ) {
        if (i > 0 && prices[i] - prices[i - 1] > 0) {
            res += prices[i] - prices[i - 1];
        }
    }
    return res;
}
 
(三）买卖股票的最佳时机 III 123   k = 2
/*
给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
*/
public int maxProfit(int[] prices) {
    if (prices == null || prices.length == 0) return 0;

    int len = prices.length;
    int[][] f = new int[len + 1][5 + 1];

    f[0][1] = 0;
    f[0][2] = f[0][3] = f[0][4] = f[0][5] = Integer.MIN_VALUE;

    for (int i = 1; i <= len; i ++ ) {
        for (int j = 1; j <= 5; j += 2) {
            f[i][j] = f[i - 1][j];
            if (i > 1 && j > 1 && f[i - 1][j - 1] != Integer.MIN_VALUE) {
                f[i][j] = Math.max(f[i][j], f[i - 1][j - 1] + prices[i - 1] - prices[i - 2]);
            } 
        }
        for (int j = 2; j <= 5; j += 2) {
            f[i][j] = f[i - 1][j - 1];
            if (i > 1 && f[i - 1][j] != Integer.MIN_VALUE) {
                f[i][j] = Math.max(f[i][j], f[i - 1][j] + prices[i - 1] - prices[i - 2]);
            }
            if (i > 1 && j > 2 && f[i - 1][j - 2] != Integer.MIN_VALUE) {
                f[i][j] = Math.max(f[i][j], f[i - 1][j - 2] + prices[i - 1] - prices[i - 2]);
            }
        }
    }

    return Math.max(Math.max(f[len][1], f[len][3]), f[len][5]);
} 
 
(四）买卖股票的最佳时机 IV 188    k = x
/*
给定一个整数数组 prices ，它的第 i 个元素 prices[i] 是一支给定的股票在第 i 天的价格。
设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
*/
public int maxProfit(int k, int[] prices) {
    if (prices == null || prices.length == 0) return 0;

    if (k > prices.length / 2) {
        int res = 0;
        for (int i = 0; i < prices.length; i ++ ) {
            if (i > 0 && prices[i] - prices[i - 1] > 0) {
                res += prices[i] - prices[i - 1];
            }
        }
        return res;
    }
    int len = prices.length;
    int[][] f = new int[len + 1][2 * k + 1 + 1];

    f[0][1] = 0;
    for (int i = 2; i <= 2 * k + 1; i ++ ) {
        f[0][i] = Integer.MIN_VALUE;
    }

    for (int i = 1; i <= len; i ++ ) {
        for (int j = 1; j <= 2 * k + 1; j += 2 ) {
            f[i][j] = f[i - 1][j];
            if (i > 1 && j > 1 && f[i - 1][j - 1] != Integer.MIN_VALUE) {
                f[i][j] = Math.max(f[i][j], f[i - 1][j - 1] + prices[i - 1] - prices[i - 2]);
            }
        }

        for (int j = 2; j <= 2 * k + 1; j += 2 ) {
            f[i][j] = f[i - 1][j - 1];
            if (i > 1 && f[i - 1][j] != Integer.MIN_VALUE) {
                f[i][j] = Math.max(f[i][j], f[i - 1][j] + prices[i - 1] - prices[i - 2]);
            }
            if (i > 1 && j > 2 && f[i - 1][j - 2] != Integer.MIN_VALUE) {
                f[i][j] = Math.max(f[i][j], f[i - 1][j - 2] + prices[i - 1] - prices[i - 2]);
            }
        }
    } 

    int res = 0;
    for (int i = 1; i<= 2 * k + 1; i += 2) {
        res = Math.max(res, f[len][i]);
    }
    return res;
}
 
(五）最佳买卖股票时机含冷冻期 309
/*

*/
 
(六）买卖股票的最佳时机含手续费 714  k = ∞
/*
给定一个整数数组 prices，其中第 i 个元素代表了第 i 天的股票价格 ；整数 fee 代表了交易股票的手续费用。
你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。
返回获得利润的最大值。
注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。
*/
public int maxProfit(int[] prices, int fee) {
    if (prices == null || prices.length == 0) return 0;

    int res = 0;
    int min = Integer.MAX_VALUE;
    for (int i = 0; i < prices.length; i ++ ) {
        if (prices[i] < min) {
            min = prices[i];
        } else if (prices[i] - min - fee > 0) {
            res += prices[i] - min - fee;
            min = prices[i] - fee; // 注意这点
        } 
    }
    return res;
} 
```

## 回文串系列

```java
409. 最长回文串 (构造最长的回文串长度)
给定一个包含大写字母和小写字母的字符串，找到通过这些字母构造成的最长的回文串。
在构造过程中，请注意区分大小写。比如 "Aa" 不能当做一个回文字符串。
    
输入:
"abccccdd"
输出:
7
解释:
我们可以构造的最长的回文串是"dccaccd", 它的长度是 7。

public int longestPalindrome(String s) {
    char[] str = s.toCharArray();
    int[] mp = new int[128];

    for (int i = 0; i < str.length; i ++ ) {
        mp[str[i]]++;
    }

    int res = 0;
    for (int cnt : mp) {
        res += cnt / 2 * 2;
        if (cnt % 2 == 1 && res % 2 == 0) {
            res++;
        }
    }
    return res;
}

5. 最长回文子串（回文子串）
给你一个字符串 s，找到 s 中最长的回文子串。

public String longestPalindrome(String ss) {
    char[] s = ss.toCharArray();
    int n = s.length;
    int x = 0;
    int y = 0;
    int len = 1;
    for (int mid = 0; mid < n; mid ++ ) {
        int i = mid;
        int j = mid;
        while (i >= 0 && j < n && s[i] == s[j]) {
            if (j - i + 1 > len) {
                len = j - i + 1;
                x = i;
                y = j;
            }
            i--;
            j++;
        }
        i = mid - 1;
        j = mid;
        while (i >= 0 && j < n && s[i] == s[j]) {
            if (j - i + 1 > len) {
                len = j - i + 1;
                x = i;
                y = j;
            }
            i--;
            j++;
        }
    }
    return ss.substring(x, y + 1);
}

647. 回文子串（求回文串数目）
给你一个字符串 s ，请你统计并返回这个字符串中 回文子串 的数目。
回文字符串 是正着读和倒过来读一样的字符串。
子字符串 是字符串中的由连续字符组成的一个序列。
具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。

public int countSubstrings(String s) {
    // 方式一
    // int res = 0;
    // char[] str = s.toCharArray();
    // for (int mid = 0; mid < str.length; mid ++ ) {
    //     int i = mid;
    //     int j = mid;
    //     while (i >= 0 && j < str.length && str[i] == str[j]) {
    //         i--;
    //         j++;
    //         res++;
    //     }
    //     i = mid - 1;
    //     j = mid;
    //     while (i >= 0 && j < str.length && str[i] == str[j]) {
    //         i--;
    //         j++;
    //         res++;
    //     }
    // }
    // return res;

    // 方式二
    char[] str = s.toCharArray();
    boolean[][] f = new boolean[str.length][str.length];
    for (boolean[] b : f) {
        Arrays.fill(b, false);
    }
    int res = 0;
    for (int j = 0; j < str.length; j ++ ) {
        for (int i = 0; i <= j; i ++ ) {
            if (str[i] == str[j] && (j - i <= 1 || f[i + 1][j - 1] == true)) {
                f[i][j] = true;
                res++;
            }
        }
    }
    return res;
}
```

## 单调栈问题总结

### 739 每日温度

```java
请根据每日 气温 列表，重新生成一个列表。对应位置的输出为：要想观测到更高的气温，至少需要等待的天数。如果气温在这之后都不会升高，请在该位置用 0 来代替。

例如，给定一个列表 temperatures = [73, 74, 75, 71, 69, 72, 76, 73]，你的输出应该是 [1, 1, 4, 2, 1, 1, 0, 0]。

提示：气温 列表长度的范围是 [1, 30000]。每个气温的值的均为华氏度，都是在 [30, 100] 范围内的整数。
    
public int[] dailyTemperatures(int[] temperatures) {
    int[] res = new int[temperatures.length];
    Stack<Integer> st = new Stack<>();
    for (int i = 0; i < temperatures.length; i ++ ) {
        while (!st.isEmpty() && temperatures[i] > temperatures[st.peek()]) {
            int preIdx = st.pop();
            res[preIdx] = i - preIdx;
        }
        st.push(i);
    }
    return res;
}    
```

### 42 接雨水

**重点：其实就是栈顶和栈顶的下一个元素以及要入栈的三个元素来接水！**

```java
当前遍历的元素i，就是凹槽右边的位置，下标为i，对应的高度为height[i]（就是图中的高度3）。

此时大家应该可以发现**其实就是栈顶和栈顶的下一个元素以及要入栈的三个元素来接水！**

那么雨水高度是 min(凹槽左边高度, 凹槽右边高度) - 凹槽底部高度，代码为：int h = min(height[st.top()], height[i]) - height[mid];

雨水的宽度是 凹槽右边的下标 - 凹槽左边的下标 - 1（因为只求中间宽度），代码为：int w = i - st.top() - 1 ;

当前凹槽雨水的体积就是：h * w。
    
public int trap(int[] height) {
    if (height == null || height.length == 0) return 0;
    int sum = 0;
    Stack<Integer> st = new Stack<>();
    for (int i = 0; i < height.length; i ++ ) {
        while (!st.isEmpty() && height[i] > height[st.peek()]) {
            int mid = st.pop();
            if (!st.isEmpty()) {
                int h = Math.min(height[i], height[st.peek()]) - height[mid];
                int w = i - st.peek() - 1;
                sum += h * w;
            }
        }
        st.push(i);
    }   
    return sum;
}
```



### 84 柱状图中最大的矩形

```java
给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。
    
public int largestRectangleArea(int[] heights) {
    int[] nHeights = new int[heights.length + 2];
    System.arraycopy(heights, 0, nHeights, 1, heights.length);

    Stack<Integer> st = new Stack<>();
    int area = 0;
    for (int i = 0; i < nHeights.length; i ++ ) {
        while (!st.isEmpty() && nHeights[i] < nHeights[st.peek()]) {
            int h = nHeights[st.pop()];
            area = Math.max(area, h * (i - st.peek() - 1));
        }
        st.push(i);
    }
    return area;
}
```

## 数组

### 1 二分查找

lc704

```java
给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/binary-search
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
    
public int search(int[] nums, int target) {
    if (nums == null || nums.length == 0) return -1;
    int l = 0;
    int r = nums.length - 1;
    while (l < r) {
        int mid = l + r >> 1;
        if (nums[mid] < target) l = mid + 1;
        else r = mid;
    }
    if (nums[l] == target) return l;
    else return -1; 
}
```

lc35

```java
给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

请必须使用时间复杂度为 O(log n) 的算法。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/search-insert-position
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
    
public int searchInsert(int[] nums, int target) {
    if (nums == null || nums.length == 0) return 0;
    int l = 0;
    int r = nums.length; // 有可能放在最后一个位置
    while (l < r) {
        int mid = l + r >> 1;
        if (nums[mid] < target) l = mid + 1;
        else r = mid;
    }
    return l;
}
```

lc34

```mysql
public int[] searchRange(int[] nums, int target) {
        if (nums == null || nums.length == 0) return new int[]{-1, -1};
        int l = 0;
        int r = nums.length - 1;
        while (l < r) {
            int mid = l + r >> 1;
            if (nums[mid] < target) l = mid + 1;
            else r = mid;
        }
        if (nums[l] != target) return new int[]{-1, -1};
        int t = l;
        r = nums.length - 1;
        while (l < r) {
            int mid = l + r + 1>> 1;
            if (nums[mid] > target) r = mid - 1;
            else l = mid;
        }
        return new int[]{t, r};
    }
```

lc69

```java
给你一个非负整数 x ，计算并返回 x 的 算术平方根 。

由于返回类型是整数，结果只保留 整数部分 ，小数部分将被 舍去 。

注意：不允许使用任何内置指数函数和算符，例如 pow(x, 0.5) 或者 x ** 0.5 

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/sqrtx
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
    
public int mySqrt(int x) {
    if (x == 0 || x == 1) return x;
    int l = 1;
    int r = x / 2;
    while (l < r) {
        int mid = l + (r - l + 1) / 2;
        if (mid > x / mid) r = mid - 1;
        else l = mid;
    }
    return l;
}
```

lc367

```java
给定一个 正整数 num ，编写一个函数，如果 num 是一个完全平方数，则返回 true ，否则返回 false 。

进阶：不要 使用任何内置的库函数，如  sqrt 。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/valid-perfect-square
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
    
public boolean isPerfectSquare(int num) {
    if (num == 1) return true;
    int l = 1;
    int r = num / 2;
    while (l < r) {
        int mid = l + r + 1 >> 1;
        if (mid > num / mid) r = mid - 1;
        else l = mid;
    }
    return num == l * l;
}
```

### 2 移除元素

lc27

```java
给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。
    
public int removeElement(int[] nums, int val) {
    int idx = 0;
    for (int i = 0; i < nums.length; i ++ ) {
        if (val != nums[i]) {
            nums[idx++] = nums[i];
        }
    }
    return idx;
}
```

lc 26.删除排序数组中的重复项

```java
给你一个 升序排列 的数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。元素的 相对顺序 应该保持 一致 。
    
public int removeDuplicates(int[] nums) {
    if (nums == null) return 0;
    if (nums.length == 0 || nums.length == 1) return 1; 
    int slow = 1;
    for (int fast = 1; fast < nums.length; fast ++ ) {
        if (nums[fast] != nums[slow - 1]) {
            nums[slow++] = nums[fast];
        }
    }
    return slow;
}
```

lc283.移动零

```java
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
    
public void moveZeroes(int[] nums) {
    int idx = 0;
    for (int i = 0; i < nums.length; i ++ ) {
        if (nums[i] != 0) {
            nums[idx++] = nums[i]; 
        }
    }
    while (idx < nums.length) {
        nums[idx++] = 0;
    }
}
```

lc844

```java
// 给定 s 和 t 两个字符串，当它们分别被输入到空白的文本编辑器后，如果两者相等，返回 true 。# 代表退格字符。

注意：如果对空文本输入退格字符，文本继续为空。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/backspace-string-compare
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
    
public boolean backspaceCompare(String s, String t) {
    return check(s).equals(check(t));
}

public String check(String ss) {
    char[] s = ss.toCharArray();
    StringBuilder str = new StringBuilder();
    for (int i = 0; i < s.length; i ++ ) {
        if (s[i] != '#') {
            str.append(s[i]);
        } else {
            if (str.length() - 1 >= 0) 
                str.deleteCharAt(str.length() - 1);
        }
    }
    return str.toString();
}
```

### 3 有序数的平方

lc977 

```java
给你一个按 非递减顺序 排序的整数数组 nums，返回 每个数字的平方 组成的新数组，要求也按 非递减顺序 排序。
    
public int[] sortedSquares(int[] nums) {
    int i = 0;
    int j = nums.length - 1;
    int[] res = new int[nums.length];
    int idx = nums.length - 1;
    while (i <= j) {
        if (nums[i] * nums[i] <= nums[j] * nums[j]) {
            res[idx--] = nums[j] * nums[j];
            j--;
        } else {
            res[idx--] = nums[i] * nums[i];
            i++;
        }
    }   
    return res;
}
```

### 4 长度最小的子数组

lc209

```java
public int minSubArrayLen(int target, int[] nums) {
    // 使用滑动窗口
    if (nums == null || nums.length == 0) return 0;
    int i = 0;
    int sum = 0;;
    int len = Integer.MAX_VALUE;
    for (int j = 0; j < nums.length; j ++ ) {
        sum += nums[j];
        while (sum > target) {
            len = Math.min(len, j - i + 1);
            sum -= nums[i++];
        }
    }
    if (len == Integer.MAX_VALUE) return 0;
    else return len;
}
```

lc904

```java
/*
你正在探访一家农场，农场从左到右种植了一排果树。这些树用一个整数数组 fruits 表示，其中 fruits[i] 是第 i 棵树上的水果 种类 。

你想要尽可能多地收集水果。然而，农场的主人设定了一些严格的规矩，你必须按照要求采摘水果：

你只有 两个 篮子，并且每个篮子只能装 单一类型 的水果。每个篮子能够装的水果总量没有限制。
你可以选择任意一棵树开始采摘，你必须从 每棵 树（包括开始采摘的树）上 恰好摘一个水果 。采摘的水果应当符合篮子中的水果类型。每采摘一次，你将会向右移动到下一棵树，并继续采摘。
一旦你走到某棵树前，但水果不符合篮子的水果类型，那么就必须停止采摘。
给你一个整数数组 fruits ，返回你可以收集的水果的 最大 数目。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/fruit-into-baskets
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/

public int totalFruit(int[] fruits) {
    if (fruits == null || fruits.length == 0) return 0;
    int i = 0;
    int res = 0;
    HashMap<Integer, Integer> map = new HashMap<>();
    for (int j = 0; j < fruits.length; j ++ ) {
        map.put(fruits[j], map.getOrDefault(fruits[j], 0) + 1);
        while (map.size() > 2) {
            map.put(fruits[i], map.get(fruits[i]) - 1);
            if (map.get(fruits[i]) == 0) map.remove(fruits[i]);
            i++;
        }
        res = Math.max(res, j - i + 1);
    }
    return res;
}
```

lc76 最小覆盖子串

```java
给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。
    
public String minWindow(String ss, String tt) {
    if (ss == null || tt == null || ss.length() == 0 || tt.length() == 0) 
        return "";

    char[] s = ss.toCharArray();
    char[] t = tt.toCharArray();
    HashMap<Character, Integer> map = new HashMap<>();
    for (int i = 0; i < t.length; i ++ ) {
        map.put(t[i], map.getOrDefault(t[i], 0) + 1);
    }

    int len = Integer.MAX_VALUE;
    int cnt = 0;
    int idx = 0;
    int i = 0;
    for (int j = 0; j < s.length; j ++ ) {
        if (map.containsKey(s[j])) {
            map.put(s[j], map.get(s[j]) - 1);
            if (map.get(s[j]) == 0) cnt++;
        }
        while (cnt == map.size()) {
            int size = j - i + 1;
            if (size < len) {
                len = size;
                idx = i;
            }
            if (map.containsKey(s[i])) {
                map.put(s[i], map.get(s[i]) + 1);
                if (map.get(s[i]) > 0) cnt--;
            }
            i++;
        }
    }
    if (len == Integer.MAX_VALUE) return "";
    else return ss.substring(idx, idx + len);
}
```

lc54

```java
给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。
    
public List<Integer> spiralOrder(int[][] matrix) {
    List<Integer> res = new ArrayList<>();
    if (matrix == null || matrix.length == 0) return new ArrayList<>(); 
    int row = matrix.length;
    int col = matrix[0].length;
    int t = 0;
    int b = row - 1;
    int l = 0;
    int r = col - 1;
    while (true) {
        for (int i = l; i <= r; i ++ ) res.add(matrix[t][i]);
        if (++t > b) break;

        for (int i = t; i <= b; i ++ ) res.add(matrix[i][r]);
        if (--r < l) break;

        for (int i = r; i >= l; i -- ) res.add(matrix[b][i]);
        if (--b < t) break;

        for (int i = b; i >= t; i -- ) res.add(matrix[i][l]);
        if (++l > r) break;
    }
    return res;
}
```



## 链表

### 1 移除链表元素

lc203 移除链表元素

```java
给你一个链表的头节点 head 和一个整数 val ，请你删除链表中所有满足 Node.val == val 的节点，并返回 新的头节点 。
    
public ListNode removeElements(ListNode head, int val) {
    ListNode dummy = new ListNode(0);
    dummy.next = head;
    ListNode cur = dummy;
    while (cur.next != null) {
        if (cur.next.val == val) {
            cur.next = cur.next.next;
        } else {
            cur = cur.next;
        }
    }
    return dummy.next;
}
```

### 2 链表相加

lc2 两数相加（首部对齐相加）

```java
public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    ListNode dummy = new ListNode(0);
    ListNode cur = dummy;
    int carry = 0;
    while (l1 != null || l2 != null || carry > 0) {
        int sum = carry;
        sum += l1 == null ? 0 : l1.val;
        sum += l2 == null ? 0 : l2.val;

        ListNode node = new ListNode(sum % 10);
        carry = sum / 10;
        cur.next = node;
        cur = cur.next;

        if (l1 != null) l1 = l1.next;
        if (l2 != null) l2 = l2.next;
    }
    return dummy.next;
}
```



lc445 两数相加ii（最后对其相加）

```mysql
给你两个 非空 链表来代表两个非负整数。数字最高位位于链表开始位置。它们的每个节点只存储一位数字。将这两数相加会返回一个新的链表。

你可以假设除了数字 0 之外，这两个数字都不会以零开头。

 

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/add-two-numbers-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        Stack<Integer> st1 = new Stack<>();
        Stack<Integer> st2 = new Stack<>();

        while (l1 != null) {
            st1.push(l1.val);
            l1 = l1.next;
        }

        while (l2 != null) {
            st2.push(l2.val);
            l2 = l2.next;
        }

        ListNode dummy = new ListNode(0);
        ListNode pointer = dummy;
        int carry = 0;
        while (!st1.isEmpty() || !st2.isEmpty() || carry > 0) {
            int sum = carry;
            sum += st1.isEmpty() ? 0 : st1.pop();
            sum += st2.isEmpty() ? 0 : st2.pop();

            ListNode node = new ListNode(sum % 10);
            carry = sum / 10;
            node.next = pointer.next;
            pointer.next = node;
        }
        return dummy.next;
    }
```

### 3 设计链表

707 设计链表

```java
get(index)：获取链表中第 index 个节点的值。如果索引无效，则返回-1。
addAtHead(val)：在链表的第一个元素之前添加一个值为 val 的节点。插入后，新节点将成为链表的第一个节点。
addAtTail(val)：将值为 val 的节点追加到链表的最后一个元素。
addAtIndex(index,val)：在链表中的第 index 个节点之前添加值为 val  的节点。如果 index 等于链表的长度，则该节点将附加到链表的末尾。如果 index 大于链表长度，则不会插入节点。如果index小于0，则在头部插入节点。


class MyLinkedList {
    ListNode dummy;
    int size;
    public MyLinkedList() {
        dummy = new ListNode(0);
        size = 0;
    }
    
    public int get(int index) {
        if (index < 0 || index >= size) return -1;
        ListNode cur = dummy.next;
        while (index-- > 0) {
            cur = cur.next;
        }
        return cur.val;
    }
    
    public void addAtHead(int val) {
        ListNode node = new ListNode(val);
        node.next = dummy.next;
        dummy.next = node;
        size++;
    }
    
    public void addAtTail(int val) {
        ListNode node = new ListNode(val);
        ListNode cur = dummy;
        while (cur.next != null) {
            cur = cur.next;
        }
        cur.next = node;
        size++;
    }
    
    public void addAtIndex(int index, int val) {
        if (index < 0 || index > size) return;
        ListNode node = new ListNode(val);
        ListNode cur = dummy;
        while (index-- > 0) {
            cur = cur.next;
        }
        node.next = cur.next;
        cur.next = node;
        size++;
    }
    
    public void deleteAtIndex(int index) {
        if (index < 0 || index >= size) return;
        ListNode cur = dummy;
        while (index-- > 0) {
            cur = cur.next;
        }
        cur.next = cur.next.next;
        size--;
    }
}

class ListNode {
    int val;
    ListNode next;
    public ListNode(int val) {
        this.val = val;
        this.next = null;
    }
    public ListNode() {}
}
```

### 4 翻转链表

```java
public ListNode reverseList(ListNode head) {
    // 递归
    // if (head == null || head.next == null) return head;
    // ListNode nHead = reverseList(head.next);
    // head.next.next = head;
    // head.next = null;
    // return nHead;

    // 迭代
    if (head == null || head.next == null) return head;

    ListNode pre = null;
    ListNode cur = head;
    ListNode tmp = null;
    while (cur != null) {
        tmp = cur.next;
        cur.next = pre;
        pre = cur;
        cur = tmp;
    }
    return pre;
}
```

### 5 两两交换链表中的节点

lc24

![24.两两交换链表中的节点1](leetcode.assets/24.两两交换链表中的节点1.png)

```java
给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。
    
public ListNode swapPairs(ListNode head) {
    ListNode dummy = new ListNode(0);
    dummy.next = head;
    ListNode cur = dummy;
    while (cur.next != null && cur.next.next != null) {
        ListNode tmp1 = cur.next;
        ListNode tmp2 = cur.next.next.next;

        cur.next = cur.next.next;
        cur.next.next = tmp1;
        cur.next.next.next = tmp2;

        cur = cur.next.next;
    }
    return dummy.next;
}
```

### 6 删除链表的倒数第N个节点

```java
给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

进阶：你能尝试使用一趟扫描实现吗？
    
public ListNode removeNthFromEnd(ListNode head, int n) {
    ListNode dummy = new ListNode(0);
    dummy.next = head;
    ListNode slow = dummy;
    ListNode fast = dummy;

    while (n-- > 0 && fast != null) {
        fast = fast.next;
    }

    fast = fast.next; // 因为要删除第n个，所以要找删除那个的前一个

    while (fast != null) {
        fast = fast.next;
        slow = slow.next;
    }

    slow.next = slow.next.next;
    return dummy.next;
}
```

### 7 链表相交

```java
面试题：02.07 给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 null 。
    
public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    ListNode curA = headA;
    ListNode curB = headB;

    while (curA != curB) {
        curA = curA == null ? headB : curA.next;
        curB = curB == null ? headA : curB.next;
    }
    return curA;
}
```

### 8 环形链表问题

lc141 环形链表

```java
给你一个链表的头节点 head ，判断链表中是否有环。
    
public boolean hasCycle(ListNode head) {
    ListNode fast = head;
    ListNode slow = head;
    while (fast != null && fast.next != null) {
        fast = fast.next.next;
        slow = slow.next;
        if (fast == slow) {
            return true;
        }
    }
    return false;
}
```

lc142 环形链表II

```java
给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。
不允许修改 链表

public ListNode detectCycle(ListNode head) {
    // 2*(x+y) = x+y+n*(y+z)
    ListNode fast = head;
    ListNode slow = head;
    while (fast != null && fast.next != null) {
        fast = fast.next.next;
        slow = slow.next;
        if (fast == slow) {
            ListNode curA = head;
            ListNode curB = slow;
            while (curA != curB) {
                curA = curA.next;
                curB = curB.next;
            }
            return curA;
        }
    }
    return null;
}
```

## 哈希表

### 1 有效的字母异位词

lc242 

```java
给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。
    
public boolean isAnagram(String s, String t) {
    char[] ss = s.toCharArray();
    char[] tt = t.toCharArray();
    int[] res = new int[26];
    for (int i = 0; i < ss.length; i ++ ) {
        res[ss[i] - 'a']++;
    }

    for (int i = 0; i < tt.length; i ++ ) {
        res[tt[i] - 'a']--;
    }

    for (int i = 0; i < 26; i ++ ) {
        if (res[i] != 0) {
            return false;
        }
    }

    return true;
}
```

lc383

```java
给你两个字符串：ransomNote 和 magazine ，判断 ransomNote 能不能由 magazine 里面的字符构成。
如果可以，返回 true ；否则返回 false 。
magazine 中的每个字符只能在 ransomNote 中使用一次。

public boolean canConstruct(String ransomNote, String magazine) {
    int[] res = new int[26];
    char[] s = ransomNote.toCharArray();
    char[] t = magazine.toCharArray();

    for (int i = 0; i < s.length; i ++ ) {
        res[s[i] - 'a']++;
    }
    for (int i = 0; i < t.length; i ++ ) {
        if (res[t[i] - 'a'] > 0) {
            res[t[i] - 'a']--;
        }
    }
    for (int i = 0; i < 26; i ++ ) {
        if (res[i] > 0) {
            return false;
        }
    } 
    return true;
}
```

lc49 字母异位词分组

```java
给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。

字母异位词 是由重新排列源单词的字母得到的一个新单词，所有源单词中的字母通常恰好只用一次。
    
public List<List<String>> groupAnagrams(String[] strs) {
    if (strs == null || strs.length == 0) return new ArrayList<>();

    Map<String, List> map = new HashMap<>();
    for (String s : strs) {
        char[] chars = s.toCharArray();
        Arrays.sort(chars);
        String key = String.valueOf(chars);
        if (!map.containsKey(key)) {
            map.put(key, new ArrayList<>());
        }
        map.get(key).add(s);
    }

    return new ArrayList(map.values());
}
```

lc438 找到字符串中的所有字母异位词

```java
//给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。
异位词 指由相同字母重排列形成的字符串（包括相同的字符串）。

public List<Integer> findAnagrams(String ss, String pp) {
    char[] s = ss.toCharArray();
    char[] p = pp.toCharArray();

    int[] scnt = new int[26];
    int[] pcnt = new int[26];
    for (int i = 0; i < p.length; i ++ ) {
        pcnt[p[i] - 'a']++;
    }
    List<Integer> res = new ArrayList<>();
    int i = 0;
    for (int j = 0; j < s.length; j ++ ) {
        scnt[s[j] - 'a']++;
        while (scnt[s[j] - 'a'] > pcnt[s[j] - 'a']) {
            scnt[s[i] - 'a']--;
            i++;
        }
        if (j - i + 1 == p.length) {
            res.add(i);
        }
    }
    return res;
}
```



### 2 两个数组的交集

lc349 两个数组的交集

```java
给定两个数组 nums1 和 nums2 ，返回 它们的交集 。输出结果中的每个元素一定是 唯一 的。我们可以 不考虑输出结果的顺序 。

public int[] intersection(int[] nums1, int[] nums2) {
    Set<Integer> set1 = new HashSet<>();
    Set<Integer> set2 = new HashSet<>();

    for (int i : nums1) {
        set1.add(i);
    }

    for (int i : nums2) {
        set2.add(i);
    }

    List<Integer> res = new ArrayList<>();
    for (int i : set1) {
        if (set2.contains(i)) {
            res.add(i);
        }
    }

    int[] ans = new int[res.size()];
    for (int i = 0; i < res.size(); i ++ ) {
        ans[i] = res.get(i);
    }

    return ans;
}
```

lc350 两个数组的交集II

```java
给你两个整数数组 nums1 和 nums2 ，请你以数组形式返回两数组的交集。返回结果中每个元素出现的次数，应与元素在两个数组中都出现的次数一致（如果出现次数不一致，则考虑取较小值）。可以不考虑输出结果的顺序。

    
public int[] intersect(int[] nums1, int[] nums2) {
    List<Integer> list = new ArrayList<>();
    for (int i : nums1) {
        list.add(i);
    }

    List<Integer> res = new ArrayList<>();
    for (int i : nums2) {
        if (list.contains(i)) {
            res.add(i);
            list.remove(Integer.valueOf(i));
        }
    }

    int[] ans = new int[res.size()];
    for (int i = 0; i < res.size(); i ++ ) {
        ans[i] = res.get(i);
    }

    return ans;
}
```



### 3 快乐数

### 4 两数之和

### 5 四数相加II

### 6 赎金信

### 7 三数之和

### 8 四数之和



## 字符串

## 双指针法

## 栈与队列

## 二叉树

## 回溯算法

## 贪心算法

## 动态规划

## 单调栈



# leetcode hot100



# 算法模板总结

## 1 八大排序

**① 冒泡排序**

```java
public void bubbleSort(int[] arr, int n) {
 	boolean flag = true;
    for (int i = 0; i < n - 1 && flag; i ++ ) {
        flag = false;
        for (int j = 0; j < n - i - 1; j ++ ) {
            if (arr[j] > arr[j + 1]) {
                swap(arr, j, j + 1);
                flag = true;
            }
        }
    }
}

public void swap(int[] arr, int i, int j) {
    int t = arr[i];
    arr[i] = arr[j];
    arr[j] = t;
}
```

**② 选择排序**

```java
public void selectSort(int[] arr, int n) {
    for (int i = 0; i < n - 1; i ++ ) {
        int min = i; // 攻擂台
        for (int j = i; j < n; j ++ ) {
            if (arr[j] < arr[min]) {
                min = j;
            }
        }
        swap(arr, i, min);
    }
}
```

**③ 快速排序**

```java
// 方式一：左右互相填充，求出标杆所在的位置，标杆左边小于它，右边大于它
public static int partition(int[] arr, int l, int r) {
    int x = arr[l];
    while (l < r) {
        while (l < r && arr[r] > x) r--;
        arr[l] = arr[r];
        while (l < r && arr[l] <= x) l++;
        arr[r] = arr[l];
    }
    arr[l] = x;
    return l;
}

public static void qSort(int[] arr, int l, int r) {
    if (l >= r) return;
    int pos = partition(arr, l, r);
    qSort(arr, l, pos - 1);
    quickSort(arr, pos + 1, r);
}

// 方式二直接一个函数
public static void quickSort(int[] arr, int l, int r) {
    if (l >= r) return;
    int i = l, j = r, x = arr[l + r >> 1];
    while (i < j) {
        while (arr[i] < x) i++;
        while (arr[j] > x) j--;
        if (i <= j) {
            swap(arr, i, j);
            i++;
            j--;
        }
    }
    if (l < j) quickSort(arr, l, j);
    if (i < r) quickSort(arr, i, r);
}
```

**④ 归并排序**

```java
public static void mergeSrt(int[] arr, int l, int r) {
        if (l >= r) return;
        int mid = l + r >> 1;
        mergeSrt(arr, l, mid);
        mergeSrt(arr, mid + 1, r);
        int[] tmp = new int[r - l + 1];
        int i = l, j = mid + 1;
        int idx = 0;
        while (i <= mid && j <= r) {
            if (arr[i] <= arr[j]) {
                tmp[idx++] = arr[i++];
            } else {
                tmp[idx++] = arr[j++];
            }
        }
        while (i <= mid) tmp[idx++] = arr[i++];
        while (j <= r) tmp[idx++] = arr[j++];

        for (int k = 0; k < idx; k ++ ) {
            arr[l + k] = tmp[k];
        }
    }
```

**⑤ 堆排序**

```java
public static void down(int[] arr, int l, int r) {
    int i = l, j = 2 * i + 1;
    while (j <= r) {
        if (j + 1 <= r && arr[j + 1] > arr[j]) {
            j++;
        }
        if (arr[j] > arr[i]) {
            swap(arr, j, i);
            i = j;
            j = 2 * i + 1;
        } else {
            break;
        }
    }
}

public static void heapSort(int[] arr, int n) {
    // 建堆
    for (int i = n / 2 - 1; i >= 0; i -- ) {
        down(arr, i, n - 1);
    }
    for (int i = n - 1; i > 0; i -- ) {
        swap(arr, 0, i);
        down(arr, 0, i - 1);
    }
}
```

**⑥ 直接插入排序**

```java
public static void insertSort(int[] arr, int n) {
    for (int i = 0; i < n; i ++ ) {
        int tmp = arr[i], j;
        for (j = i - 1; j >= 0 && arr[j] > tmp; j -- ) {
            arr[j + 1] = arr[j];
        }
        arr[j + 1] = tmp;
    }
}
```

**⑦ 折半插入排序**

```java
public static void halfInsertSort(int[] arr, int n) {
    for (int i = 0; i < n; i ++ ) {
        int tmp = arr[i], j;
        int l = 0, r = i;
        while (l < r) {
            int mid = l + r >> 1;
            if (arr[mid] <= tmp) l = mid + 1;
            else r = mid;
        }
        for (j = i - 1; j >= l; j -- ) {
            arr[j + 1] = arr[j];
        }
        arr[j + 1] = tmp;
    }
}
```

**⑧ 希尔排序**

```java
public static void shellSort(int[] arr, int n) {
    for (int dk = n / 2; dk >= 1; dk /= 2 ) {
        for (int i = dk; i < n; i ++ ) {
            if (arr[i - dk] > arr[i]) {
                int tmp = arr[i], j;
                for (j = i - dk; j >= 0 && arr[j] > tmp; j -= dk ) {
                    arr[j + dk] = arr[j];
                }
                arr[j + dk] = tmp;
            }
        }
    }
}
```

## 2 二分模板

```java
// 寻找第一个x
int l = 0, r = n - 1;
while (l < r) {
    int mid = l + r >> 1;
    if (arr[mid] < x) l = mid + 1;
    else r = mid;
}
if (arr[l] != x) return -1;
return l;  // l和r都是一样的，因为l = r跳出循环


// 寻找最后一个x
int l = 0, r = n - 1;
while (l < r) {
    int mid = l + r + 1 >> 1; (注意改边界问题)
    if (arr[mid] > x) r = nid - 1;
    else l = mid;
}
if (arr[mid] != x) return -1;
return l;
```

