2/17 Meet with Maria. TODOs for next week:

- [x] clarify unknown formulas and algorithms in the [paper](https://jov.arvojournals.org/article.aspx?articleid=2772700)
    - central differenct filter: use convolution to calculate derivative (velocity).
    - max amplitude = 35: human eyes' movement to oneside is about 30-35 degs max.
    - fine grid search: basically a grid search that explore all posibility and find the parameteres that produce optimal result.
    - saccade direction: not explicitly mentioned in the [paper](https://jov.arvojournals.org/article.aspx?articleid=2772700), but it's one dimentional, either horizontal or vertical.
- [x] format [source code](https://eeweb.engineering.nyu.edu/iselesni/eye-movement/): add comments, remove unused codes, ...
    - [sparse matrix](https://phys.libretexts.org/Bookshelves/Mathematical_Physics_and_Pedagogy/Computational_Physics_(Chong)/08%3A_Sparse_Matrices/8.02%3A_Sparse_Matrix_Formats#:~:text=in%20array%20format!-,8.2.2%20Diagonal%20Storage%20(DIA),-The%20Diagonal%20Storage) (scipy.sparse.spdiags)

2/28 MeetingNote
- Getting some anti-saccades data

4/18 MeetingNote
- [] seperate parse data and plot data
- [] create file for read in sample data and anti-saccade data
- [] turn moving average into low pass filter