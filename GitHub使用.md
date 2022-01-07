### 1、从 github 或 gitee 下载代码
git clone 项目路径 
例如 ：git clone https://gitee.com/tus_xxw/CS-Notes.git

### 2、上传我们的代码到github 或者gitee

​	全局设置

​	git config --global user.name "koiman"

​	git config --global user.email "abc@qq.com"



git status  //这个是查看git当时的状态

git add .  //添加所有修改的文件
git commit -m '这里面是提交的注释'  //提交代码
git push // 推送代码到gitee 或者github 
git pull //从服务端gitee 或者 github拉取代码

### 3、演示合作开发

修改自己的代码之前先git pull获取今天更新的代码

如果自己修改本地代码之前不git pull更新，然后直接修改代码，再次git commit之后会发生冲突

此时可以修改冲突文件再次提交

![image-20220107161045154](C:\Users\tesseract\AppData\Roaming\Typora\typora-user-images\image-20220107161045154.png)

![image-20220107161000193](C:\Users\tesseract\AppData\Roaming\Typora\typora-user-images\image-20220107161000193.png)

### 4、个人新建仓库并往上面推代码

① git init //本地初始化git仓库
② git add .
③ git commit -m '...'
④ git remote add origin https://gitee.com...  //添加远程仓库地址
⑤ git push   // 如果说没推上去 就用 git push --set-upstream origin master

git push -u origin master

### 5、添加一个新的远程仓库

git remote add github https://github.com/xiaoweix/myCode.git
git remote -v // 查看自己有哪些远程仓库

### 6、误删 回退到上一个版本


git reset --hard HEAD^
git reset --hard commit号 //回退到指定commit版本 使用 git log 查看commit号

### 7、分支
git checkout test //切换分支