from model.parameters.DefaultParameters import DefaultParameters

class DefaultParameterFactory:
    def __init__(self):
        self.parameters = DefaultParameters()

    def create_default_parameters(self, parameters={}):
        if "stress" in parameters:
            self.parameters.default_stress = self.create_stress_parameter_set(parameters["stress"])
        if "aversion" in parameters:
            self.parameters.default_aversion = self.create_aversion_parameter_set(parameters["aversion"])
        if "urge_to_escape" in parameters:
            self.parameters.default_urge_to_escape = self.create_urge_to_escape_parameter_set(parameters["urge_to_escape"])
        if "suicide_history" in parameters:
            self.parameters.default_suicide_history = self.create_suicide_history_parameter_set(parameters["suicide_history"])
        if "suicidal_thought" in parameters:
            self.parameters.default_suicidal_thought = self.create_suicidal_thought_parameter_set(parameters["suicidal_thought"])
        if "escape_behavior" in parameters:
            self.parameters.default_escape_behavior = self.create_escape_behavior_parameter_set(parameters["escape_behavior"])
        if "external_strategy" in parameters:
            self.parameters.default_external_strategy = self.create_external_strategy_parameter_set(parameters["external_strategy"])
        if "internal_strategy" in parameters:
            self.parameters.default_internal_strategy = self.create_internal_strategy_parameter_set(parameters["internal_strategy"])
        if "burdensomeness" in parameters:
            self.parameters.default_burdensomeness = self.create_burdensomeness_parameter_set(parameters["burdensomeness"])
        return self.parameters

    def create_stress_parameter_set(self, params):
        stress_params = self.parameters.default_stress
        if "baseline" in params:
            stress_params.baseline = params["baseline"]
        if "decay" in params:
            stress_params.decay = params["decay"]
        if "impulse_rate" in params:
            stress_params.impulse_rate = params["impulse_rate"]
        if "impulse_strength" in params:
            stress_params.impulse_strength = params["impulse_strength"]
        if "morning_impulse" in params:
            stress_params.morning_impulse = params["morning_impulse"]
        if "alpha" in params:
            stress_params.alpha = params["alpha"]
        if "beta" in params:
            stress_params.beta = params["beta"]
        if "gamma" in params:
            stress_params.gamma = params["gamma"]
        if "sigma" in params:
            stress_params.sigma = params["sigma"]
        return stress_params

    def create_aversion_parameter_set(self, params):
        aversion_params = self.parameters.default_aversion
        if "feedback" in params:
            aversion_params.feedback = params["feedback"]
        if "carrying_capacity" in params:
            aversion_params.carrying_capacity = params["carrying_capacity"]
        if "S_weight" in params:
            aversion_params.S_weight = params["S_weight"]
        if "T_weight" in params:
            aversion_params.T_weight = params["T_weight"]
        if "X_weight" in params:
            aversion_params.X_weight = params["X_weight"]
        if "I_weight" in params:
            aversion_params.I_weight = params["I_weight"]
        if "B_weight" in params:
            aversion_params.B_weight = params["B_weight"]
        if "c_weight" in params:
            aversion_params.c_weight = params["c_weight"]
        return aversion_params

    def create_urge_to_escape_parameter_set(self, params):
        urge_params = self.parameters.default_urge_to_escape
        if "feedback" in params:
            urge_params.feedback = params["feedback"]
        if "A_weight" in params:
            urge_params.A_weight = params["A_weight"]
        if "M_weight" in params:
            urge_params.M_weight = params["M_weight"]
        if "C_weight" in params:
            urge_params.C_weight = params["C_weight"]
        return urge_params

    def create_suicide_history_parameter_set(self, params):
        history_params = self.parameters.default_suicide_history
        if "decay" in params:
            history_params.decay = params["decay"]
        return history_params

    def create_suicidal_thought_parameter_set(self, params):
        suicidal_params = self.parameters.default_suicidal_thought
        if "feedback" in params:
            suicidal_params.feedback = params["feedback"]
        if "sig_middle" in params:
            suicidal_params.sig_middle = params["sig_middle"]
        if "sig_steepness" in params:
            suicidal_params.sig_steepness = params["sig_steepness"]
        return suicidal_params

    def create_escape_behavior_parameter_set(self, params):
        escape_params = self.parameters.default_escape_behavior
        if "feedback" in params:
            escape_params.feedback = params["feedback"]
        if "sig_middle" in params:
            escape_params.sig_middle = params["sig_middle"]
        if "sig_steepness" in params:
            escape_params.sig_steepness = params["sig_steepness"]
        return escape_params

    def create_external_strategy_parameter_set(self, params):
        external_params = self.parameters.default_external_strategy
        if "feedback" in params:
            external_params.feedback = params["feedback"]
        if "carrying_capacity" in params:
            external_params.carrying_capacity = params["carrying_capacity"]
        if "A_weight" in params:
            external_params.A_weight = params["A_weight"]
        if "U_weight" in params:
            external_params.U_weight = params["U_weight"]
        return external_params

    def create_internal_strategy_parameter_set(self, params):
        internal_params = self.parameters.default_internal_strategy
        if "feedback" in params:
            internal_params.feedback = params["feedback"]
        if "carrying_capacity" in params:
            internal_params.carrying_capacity = params["carrying_capacity"]
        if "A_weight" in params:
            internal_params.A_weight = params["A_weight"]
        if "U_weight" in params:
            internal_params.U_weight = params["U_weight"]
        return internal_params

    def create_burdensomeness_parameter_set(self, params):
        burden_params = self.parameters.default_burdensomeness
        if "neighbors" in params:
            burden_params.neighbors = params["neighbors"]
        if "neighbor_ws" in params:
            burden_params.neighbor_ws = params["neighbor_ws"]
        if "feedback" in params:
            burden_params.feedback = params["feedback"]
        if "A_weight" in params:
            burden_params.A_weight = params["A_weight"]
        if "I_weight" in params:
            burden_params.I_weight = params["I_weight"]
        if "B_lonely" in params:
            burden_params.B_lonely = params["B_lonely"]
        return burden_params
