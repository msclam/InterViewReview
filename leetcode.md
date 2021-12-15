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
题目一： 从前序与中序遍历序列构造二叉树
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

题目二： 从后序与中序遍历序列构造二叉树
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
    
class Solution {
    public double myPow(double x, int n) {
        long m = n;
        if (m < 0) {
            x = 1 / x;
            m = -m;
        }
        double res = 1.0;
        while (m != 0) {
            if (m % 2 != 0) res *= x;
            x = x * x;
            m >>= 1; 
        }
        return res;
    }
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

 

## 34 二叉树中和为某一值得路径 （路径总和II）

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



## 34 路径总和I

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

// hash法构建新链表
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
        int res = nums[0];
        int cnt = 0;
        for (int i = 0; i < nums.length; i ++ ) {
            if (cnt == 0) {
                res = nums[i];
                cnt = 1;
            } else {
                if (res == nums[i]) cnt++;
                else cnt--;
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



# Carl习题集

