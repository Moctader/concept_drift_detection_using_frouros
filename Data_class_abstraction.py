from abc import ABC, abstractmethod
from typing import Dict
from dataclasses import dataclass, field
from typing import List, Optional
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseMetrics(ABC):
    @abstractmethod
    def update_metrics(self, **kwargs):
        """Update metrics with provided keyword arguments."""
        pass

    @abstractmethod
    def get_latest_metrics(self) -> Dict:
        """Retrieve the most recent metrics."""
        pass



@dataclass
class Concept_Drift_Metrics(BaseMetrics):
    steps: List[int] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)
    precisions: List[float] = field(default_factory=list)
    recalls: List[float] = field(default_factory=list)
    f1_scores: List[float] = field(default_factory=list)
    drift_points: List[int] = field(default_factory=list)


    def update_metrics(
        self,
        step: int,
        accuracy: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        f1_score: Optional[float] = None,
        drift_point: Optional[int] = None,
        
    ):
        self.steps.append(step)
        if accuracy is not None:
            self.accuracies.append(accuracy)
            logger.info(f"Updated accuracy: {accuracy}")
        if precision is not None:
            self.precisions.append(precision)
            logger.info(f"Updated precision: {precision}")
        if recall is not None:
            self.recalls.append(recall)
            logger.info(f"Updated recall: {recall}")
        if f1_score is not None:
            self.f1_scores.append(f1_score)
            logger.info(f"Updated F1 score: {f1_score}")
        if drift_point is not None:
            self.drift_points.append(drift_point)
            logger.info(f"Updated drift point: {drift_point}")



    def get_latest_metrics(self) -> dict:
        """Retrieve the most recent metrics."""
        latest_metrics = {
            'accuracy': self.accuracies[-1] if self.accuracies else None,
            'precision': self.precisions[-1] if self.precisions else None,
            'recall': self.recalls[-1] if self.recalls else None,
            'f1_score': self.f1_scores[-1] if self.f1_scores else None,
            'drift_point': self.drift_points[-1] if self.drift_points else None,

        }
        logger.info(f"Latest metrics: {latest_metrics}")
        return latest_metrics
    

@dataclass
class Target_drift_Metircs(BaseMetrics):
    p_values: List[float] = field(default_factory=list)

    def update_metrics(self, p_value: float):
        self.p_values.append(p_value)
        logger.info(f"Updated p-value: {p_value}")

    def get_latest_metrics(self) -> dict:
        """Retrieve the most recent metrics."""
        latest_metrics = {
            'p_value': self.p_values[-1] if self.p_values else None,
        }
        logger.info(f"Latest metrics: {latest_metrics}")
        return latest_metrics
    


@dataclass
class Profit_loss_Metrics(BaseMetrics):
    profits: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)

    def update_metrics(self, profit: float, loss: float):
        self.profits.append(profit)
        self.losses.append(loss)
        logger.info(f"Updated profit: {profit}")
        logger.info(f"Updated loss: {loss}")

    def get_latest_metrics(self) -> dict:
        """Retrieve the most recent metrics."""
        latest_metrics = {
            'profit': self.profits[-1] if self.profits else None,
            'loss': self.losses[-1] if self.losses else None,
        }
        logger.info(f"Latest metrics: {latest_metrics}")
        return latest_metrics