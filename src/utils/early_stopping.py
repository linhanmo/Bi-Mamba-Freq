from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int
    mode: str = "min"
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.best_score = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        improved = False
        if self.best_score is None:
            improved = True
        elif self.mode == "min":
            improved = value < (self.best_score - self.min_delta)
        else:
            improved = value > (self.best_score + self.min_delta)

        if improved:
            self.best_score = value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.should_stop = True
        return improved

