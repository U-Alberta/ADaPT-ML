"""
just config things
"""
import sys
from configparser import ConfigParser, NoSectionError
from utils import CONFIG_PATH

LM_OPTIMIZER_SETTINGS = {
    'sgd': {
        'lr': 0.01,
        'l2': 1e-5,                     # keep this between 0 and 0.1
        'lr_scheduler': 'constant'
    },
    'adam': {
        'lr': 0.001,
        'l2': 1e-5,
        'lr_scheduler': 'constant'
    },
    'adamax': {
        'lr': 0.002,
        'l2': 1e-5,
        'lr_scheduler': 'constant'
    }
}

"""
Penalty: Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 
 penalties. ‘elasticnet’ is only supported by the ‘saga’ solver. If ‘none’ (not supported by the liblinear solver), no 
 regularization is applied.

Dual: Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. 
 Prefer dual=False when n_samples > n_features.

Intercept Scaling: Useful only when the solver ‘liblinear’ is used and self.fit_intercept is set to True. In this case,
 x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equal to intercept_scaling is 
 appended to the instance vector. The intercept becomes intercept_scaling * synthetic_feature_weight. 
 Note! the synthetic feature weight is subject to l1/l2 regularization as all other features. To lessen the effect of 
 regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.

Random State: Used when solver == ‘sag’, ‘saga’ or ‘liblinear’ to shuffle the data.

L1 Ratio: The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. Only used if penalty='elasticnet'. 
 Setting l1_ratio=0 is equivalent to using penalty='l2', while setting l1_ratio=1 is equivalent to using penalty='l1'. 
 For 0 < l1_ratio <1, the penalty is a combination of L1 and L2.
"""
MLOGIT_PARAMETERS = {
    'newton-cg': {
        'penalty': 'l2'
    },
    'lbfgs': {
        'penalty': 'l2',
        'verbose': 10000
    },
    'liblinear': {
        'penalty': 'l1',
        'fit_intercept': True,
        'intercept_scaling': 1,
        'random_state': 42,
        'multi_class': 'ovr',
        'verbose': 10000
    },
    'sag': {
        'penalty': 'l2',
        'random_state': 42,
    },
    'saga': {
        'penalty': 'elasticnet',
        'random_state': 42,
        'l1_ratio': 0.7
    }
}


def read_config(section: str) -> dict:
    """
    Reads config.cfg at the specified section
    http://www.postgresqltutorial.com/postgresql-python/connect/

    :param section: string of the section name
    :return:
    """
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(CONFIG_PATH)
    p_dict = {}
    try:
        params = parser.items(section)
    except NoSectionError:
        sys.exit('Section {0} not found in {1}'.format(section, CONFIG_PATH))
    for param in params:
        converted = _convert(param[1])
        p_dict[param[0]] = converted
    return p_dict


def _convert(param: str):
    """
    Converts parameters from strings to their correct data types

    :param param: string parameter from the config file
    :return:
    """
    try:
        return int(param)
    except ValueError:
        try:
            return float(param)
        except ValueError:
            if param.lower() in ['true', 'on', 'yes', 't', 'y']:
                return True
            elif param.lower() in ['false', 'off', 'no', 'f', 'n']:
                return False
            elif param.lower() in ['none', 'null']:
                return None
            else:
                return param


def _complete_params(section: str, p_dict: dict) -> dict:
    """
    Fills in the rest of the parameters for the models that shouldn't really be messed with too much
    :param section:
    :param p_dict:
    :return:
    """
    if section == 'label_model':
        p_dict.update(LM_OPTIMIZER_SETTINGS[p_dict['optimizer']])
    elif section == 'mlogit':
        p_dict.update(MLOGIT_PARAMETERS[p_dict['solver']])
    return p_dict
