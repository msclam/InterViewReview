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



# Carl习题集

