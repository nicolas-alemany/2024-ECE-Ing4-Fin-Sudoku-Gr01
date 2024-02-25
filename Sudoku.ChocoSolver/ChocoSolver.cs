using Sudoku.Shared;
using org.chocosolver.solver;
    
namespace Sudoku.ChocoSolver;

public class ChocoSolver : ISudokuSolver {

        
    public SudokuGrid Solve(SudokuGrid s) {

        var sudokuToSolve = s.CloneSudoku();

        Model model = new Model();

        return s;
    }




}
