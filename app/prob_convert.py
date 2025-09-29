# ==============================================================================
# Probability Conversion Utilities for Crash Prediction
# ==============================================================================
# Purpose: Convert model probabilities to calibrated real-world probabilities
#          using smoothed isotonic regression and base rate correction
#
# Input Files:
#   - ../models/probability_calibrator.pkl: Pre-trained calibration model
#
# Output:
#   - Calibrated real-world probability values and formatting utilities
# ==============================================================================

# ================= IMPORTS =================

import pickle
import logging
import numpy as np

# ================= CONFIGURATION =================

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
    )
logger = logging.getLogger(__name__)

# ================= MAIN CONVERTER CLASS =================

class ProbabilityConverter:
    """Converts model probabilities to real-world probabilities"""

    def __init__(self, calibrator_path='../models/probability_calibrator.pkl'):
        """Initialize converter with calibrator path and load components"""

        # Initialize vars
        self.calibrator_path = calibrator_path
        self.isotonic_calibrator = None
        self.real_base_rate = None
        self.training_base_rate = None

        # Operational safety parameters
        self.max_realistic = 0.01    # Cap final probability at 1% per hour
        self._eps = 1e-12            # Epsilon for numerical stability
        
        # Pre-computed correction factor (set during loading)
        self._correction_factor = None

        # Load calibrator 
        self.load_calibrator()

    def load_calibrator(self):
        """Load pre-trained calibrator and compute correction factors"""

        # Load calibrator 
        with open(self.calibrator_path, 'rb') as f:
                calibrator_data = pickle.load(f)

        # Extract calibrator components
        self.isotonic_calibrator = calibrator_data['isotonic_calibrator']
        self.real_base_rate = calibrator_data['real_base_rate']
        self.training_base_rate = calibrator_data['training_base_rate']
        
        # Pre-compute base rate correction factor for efficiency
        self._correction_factor = (self.real_base_rate / self.training_base_rate) * \
                                ((1 - self.training_base_rate) / (1 - self.real_base_rate))

    def isotonic_predict(self, x):
        """Apply smoothed isotonic calibration with bounds checking"""
        
        # Simple transform with bounds
        result = self.isotonic_calibrator.transform([x])[0]

        return float(np.clip(result, 0.0, 1.0))

    def convert(self, model_prob):
        """Convert model probability to real-world probability"""

        # Apply smoothed isotonic calibration
        calibrated_prob = self.isotonic_predict(model_prob)
        
        # Apply base rate correction 
#================= CHANGE THIS WHEN BASE RATE CORRECTION IS READY =================
        corrected = calibrated_prob # delete
        #corrected = self.apply_base_rate_correction(calibrated_prob) # uncomment
        
        # Apply final safety cap
        final = min(corrected, self.max_realistic)

        return final

    def apply_base_rate_correction(self, calibrated_prob):
        """Apply base rate correction using pre-computed correction factor"""

        # Ensure valid probability range for odds calculation
        p = np.clip(calibrated_prob, self._eps, 1.0 - self._eps)
        
        # Convert to odds, apply correction, convert back to probability
        odds_training = p / (1 - p)
        odds_real = odds_training * self._correction_factor
        prob_real = odds_real / (1 + odds_real)
        
        # Cap at max
        return float(np.clip(prob_real, 0.0, 1.0))


# ================= FORMATTING UTILITIES =================

def format_percent(probability):
    """Format probability as friendly percentage with adaptive precision"""

    if probability < 0.0001:
        return f"{probability*100:.6f}%"
    elif probability < 0.001:
        return f"{probability*100:.5f}%"
    elif probability < 0.01:
        return f"{probability*100:.3f}%"
    else:
        return f"{probability*100:.2f}%"

# ================= CONVENIENCE FUNCTIONS =================

def load_converter(calibrator_path='../models/probability_calibrator.pkl'):
    """Load and return a configured probability converter instance"""

    return ProbabilityConverter(calibrator_path)

def convert_prob(model_prob, converter):
    """Convert probability using provided converter instance"""

    return converter.convert(model_prob)

