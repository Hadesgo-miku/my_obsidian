这是一种将任意可逆矩阵，分解成标准正交矩阵和可逆上三角矩阵的方法

设A为域上n阶可逆方阵，则存在分解$A=QR$，其中Q为[[酉矩阵]]，R为可逆上三角矩阵
这种分解实际上是对施密特正交化过程的一个记录


求酉矩阵 $\boldsymbol{Q}$ 及上三角形矩阵 $\boldsymbol{R}$ ，使 $\boldsymbol{A}=\boldsymbol{Q} \boldsymbol{R}$ 。
解 设 $\boldsymbol{\alpha}_1, \boldsymbol{\alpha}_2, \boldsymbol{\alpha}_3, \boldsymbol{\alpha}_4 \in \mathbb{F}^4$ 依次是 $\boldsymbol{A}$ 的第 $1,2,3,4$ 列向量，它们线性无关．取

$$
\begin{aligned}
& \boldsymbol{\beta}_1=\boldsymbol{\alpha}_1=\left(\begin{array}{l}
1 \\
1 \\
0 \\
0  
\end{array}\right), \boldsymbol{\beta}_2=\boldsymbol{\alpha}_2-\frac{\left(\boldsymbol{\alpha}_2, \boldsymbol{\beta}_1\right)}{\left(\boldsymbol{\beta}_1, \boldsymbol{\beta}_1\right)} \boldsymbol{\beta}_1=\boldsymbol{\alpha}_2-\frac{1}{2} \boldsymbol{\beta}_1=\left(\begin{array}{r}
\frac{1}{2} \\
-\frac{1}{2} \\
1 \\
0
\end{array}\right), \\
& \boldsymbol{\beta}_3=\boldsymbol{\alpha}_3-\frac{\left(\boldsymbol{\alpha}_3, \boldsymbol{\beta}_1\right)}{\left(\boldsymbol{\beta}_1, \boldsymbol{\beta}_1\right)} \boldsymbol{\beta}_1-\frac{\left(\boldsymbol{\alpha}_3, \boldsymbol{\beta}_2\right)}{\left(\boldsymbol{\beta}_2, \boldsymbol{\beta}_2\right)} \boldsymbol{\beta}_2=\boldsymbol{\alpha}_3+\frac{1}{2} \boldsymbol{\beta}_1+\frac{1}{3} \boldsymbol{\beta}_2=\left(\begin{array}{r}
-\frac{1}{3} \\
\frac{1}{3} \\
\frac{1}{3} \\
1
\end{array}\right), \\
& \boldsymbol{\beta}_4=\boldsymbol{\alpha}_4-\frac{\left(\boldsymbol{\alpha}_4, \boldsymbol{\beta}_1\right)}{\left(\boldsymbol{\beta}_1, \boldsymbol{\beta}_1\right)} \boldsymbol{\beta}_1-\frac{\left(\boldsymbol{\alpha}_4, \boldsymbol{\beta}_2\right)}{\left(\boldsymbol{\beta}_2, \boldsymbol{\beta}_2\right)} \boldsymbol{\beta}_2-\frac{\left(\boldsymbol{\alpha}_4, \boldsymbol{\beta}_3\right)}{\left(\boldsymbol{\beta}_3, \boldsymbol{\beta}_3\right)} \boldsymbol{\beta}_3=\left(\begin{array}{r}
1 \\
-1 \\
-1 \\
1
\end{array}\right),
\end{aligned}
$$

这样简单移项得到$\alpha_{1}=\beta_{1},\alpha_{2}=\beta_{2}+\frac{1}{2}\beta_{1},\alpha_{3}=\beta_{3}-\frac{1}{2}\beta_{1}-\frac{1}{3}\beta_{2}$,
就能得到
$(\alpha_{1},\alpha_{2},\alpha_{3},\alpha_{4})=(\beta_{1},\beta_{2},\beta_{3},\beta_{4})M$
这里M为上三角阵
再对$(\beta_{1},\beta_{2},\beta_{3},\beta_{4})$进行归一化，$(\beta_{1},\beta_{2},\beta_{3},\beta_{4})=(\eta_{1},\eta_{2},\eta_{3},\eta_{4})D$
其中D为对角阵

则取$Q=(\eta_{1},\eta_{2},\eta_{3},\eta_{4})，R=DM$


