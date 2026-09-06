from pinn_audit.extract import _parse_table, extract_collocation_ablation


def test_table_parser_handles_mean_and_std():
    table = r"""
    \begin{tabular}{cc|cc}
    L2RE & Name & Family & Family \\
    -- & & PINN & LAAF \\
    \multirow{1}{*}{Heat} & 2d-CG & 3.64E-2(8.82E-3) & \textbf{2.39E-2(1.39E-3)} \\
    \end{tabular}
    """
    result = _parse_table(table, "appendix_table")
    assert list(result["method"]) == ["PINN", "LAAF"]
    assert result.iloc[1]["mean"] == 2.39e-2
    assert result.iloc[1]["std"] == 1.39e-3


def test_collocation_parser_handles_wrapped_rows():
    tex = r"""
    \begin{tabular}{cc|cccc}
    \multicolumn{2}{c|}{L2RE} & Burgers1d & GS & Heat2d-CG & Poisson2d-C \\
    \multirow{4}{*}{PINN} & 512 & 4.59E-1(8.36E-2) & 2.46E-1(1.09E-1)
      & 4.31E-1(6.57E-2) & 3.15E-2(4.04E-3) \\
    \end{tabular}
    \caption{x}
    \label{ab-bsize}
    """
    result = extract_collocation_ablation(tex)
    assert len(result) == 4
    assert set(result["case"]) == {"Burgers1d", "GS", "Heat2d-CG", "Poisson2d-C"}
    assert set(result["method"]) == {"PINN"}
