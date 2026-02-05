class InspectionReport:
    """
    Lightweight container for dataset inspection results.
    Presentation-only: no data logic lives here.
    """

    def __init__(
        self,
        shape: dict,
        risk: dict,
        suggestions: list[dict],
        consistency_issues: list[str],
    ):
        self.shape = shape
        self.risk = risk
        self.suggestions = suggestions
        self.consistency_issues = consistency_issues

    # --------------------
    # Human-readable output
    # --------------------
    def show(self) -> None:
        print("\nrapidds — dataset inspection")
        print("-" * 30)

        print(f"\nRows: {self.shape['rows']}")
        print(f"Columns: {self.shape['columns']}")
        print(f"Overall risk: {self.risk['overall_risk'].upper()}")

        if self.consistency_issues:
            print("\nCONSISTENCY ISSUES:")
            for issue in self.consistency_issues:
                print(f" • {issue}")

        if not self.suggestions:
            print("\nNo major issues detected.")
            return

        for level in ["high", "medium", "low"]:
            items = [s for s in self.suggestions if s["level"] == level]
            if items:
                print(f"\n{level.upper()} PRIORITY:")
                for item in items:
                    print(f" • {item['message']}")

        print("\nReview suggestions before applying any cleaning steps.")

    # --------------------
    # Markdown export
    # --------------------
    def to_markdown(self) -> str:
        lines = [
            "## rapidds — dataset inspection",
            "",
            f"- Rows: {self.shape['rows']}",
            f"- Columns: {self.shape['columns']}",
            f"- Overall risk: **{self.risk['overall_risk'].upper()}**",
            "",
        ]

        if self.consistency_issues:
            lines.append("### Consistency issues")
            for issue in self.consistency_issues:
                lines.append(f"- {issue}")
            lines.append("")

        if not self.suggestions:
            lines.append("No major issues detected.")
            return "\n".join(lines)

        for level in ["high", "medium", "low"]:
            items = [s for s in self.suggestions if s["level"] == level]
            if items:
                lines.append(f"### {level.capitalize()} priority")
                for item in items:
                    lines.append(f"- {item['message']}")
                lines.append("")

        return "\n".join(lines)
