''' 
Artificial Intelligence Techniques SL
artelnics@artelnics.com

Your model has been exported to this python file.
You can manage it with the 'NeuralNetwork' class.
Example:
 
	model = NeuralNetwork()
	sample = [input_1, input_2, input_3, input_4, ...]
	outputs = model.calculate_outputs(sample)
 
 
Inputs Names: 
	0) long_pos_cob
	1) prismatic_coef
	2) length_displacement
	3) beam_draught
	4) length_beam
	5) froude

You can predict with a batch of samples using calculate_batch_output method	 
IMPORTANT: input batch must be <class 'numpy.ndarray'> type
Example_1:
	model = NeuralNetwork()
	input_batch = np.array([[1, 2], [4, 5]])
	outputs = model.calculate_batch_output(input_batch)
Example_2:
	input_batch = pd.DataFrame( {'col1': [1, 2], 'col2': [3, 4]})
	outputs = model.calculate_batch_output(input_batch.values)
''' 
import numpy as np
import pandas as pd

class NeuralNetwork:

	def __init__(self):
		self.inputs_number = 6
		self.input_names = ['long_pos_cob', 'prismatic_coef', 'length_displacement', 'beam_draught', 'length_beam', 'froude']

	@staticmethod
	def Linear(x):
		return x

	@staticmethod
	def HyperbolicTangent(x):
		return np.tanh(x)

	def calculate_outputs(self, inputs):
		long_pos_cob = inputs[0]
		prismatic_coef = inputs[1]
		length_displacement = inputs[2]
		beam_draught = inputs[3]
		length_beam = inputs[4]
		froude = inputs[5]

		scaled_long_pos_cob = (long_pos_cob+2.381818295)/1.51321888
		scaled_prismatic_coef = (prismatic_coef-0.5641363859)/0.02329002135
		scaled_length_displacement = (length_displacement-4.788636684)/0.2530569732
		scaled_beam_draught = (beam_draught-3.936818123)/0.5481929779
		scaled_length_beam = (length_beam-3.206818104)/0.2479983717
		scaled_froude = (froude-0.287499994)/0.1009422243
		dense2d_layer_1_output_0 = self.HyperbolicTangent( -0.109461 + (0.0116255*scaled_long_pos_cob) + (-0.0127097*scaled_prismatic_coef) + (-0.00536029*scaled_length_displacement) + (0.00251012*scaled_beam_draught) + (-0.000433975*scaled_length_beam) + (-0.0322715*scaled_froude) )
		dense2d_layer_1_output_1 = self.HyperbolicTangent( -0.0761578 + (-0.00105008*scaled_long_pos_cob) + (-0.00723293*scaled_prismatic_coef) + (-0.000235623*scaled_length_displacement) + (-0.000268065*scaled_beam_draught) + (2.05285e-05*scaled_length_beam) + (-0.0267205*scaled_froude) )
		dense2d_layer_1_output_2 = self.HyperbolicTangent( 0.29369 + (0.00287934*scaled_long_pos_cob) + (0.0541758*scaled_prismatic_coef) + (-0.0813595*scaled_length_displacement) + (0.0065767*scaled_beam_draught) + (-0.118629*scaled_length_beam) + (0.0377366*scaled_froude) )
		dense2d_layer_1_output_3 = self.HyperbolicTangent( 0.0141507 + (0.00494013*scaled_long_pos_cob) + (0.00028475*scaled_prismatic_coef) + (0.0109104*scaled_length_displacement) + (0.00269852*scaled_beam_draught) + (0.00113133*scaled_length_beam) + (0.00930119*scaled_froude) )
		dense2d_layer_1_output_4 = self.HyperbolicTangent( -0.12185 + (0.00667074*scaled_long_pos_cob) + (-0.0218321*scaled_prismatic_coef) + (-0.0114995*scaled_length_displacement) + (-0.00795459*scaled_beam_draught) + (0.00395799*scaled_length_beam) + (-0.0483556*scaled_froude) )
		dense2d_layer_1_output_5 = self.HyperbolicTangent( -0.0207175 + (0.00322004*scaled_long_pos_cob) + (-0.00243719*scaled_prismatic_coef) + (-0.0105572*scaled_length_displacement) + (-0.00903828*scaled_beam_draught) + (0.00172959*scaled_length_beam) + (-0.00605992*scaled_froude) )
		dense2d_layer_1_output_6 = self.HyperbolicTangent( 0.315469 + (0.00426721*scaled_long_pos_cob) + (-0.0332213*scaled_prismatic_coef) + (0.0755265*scaled_length_displacement) + (-0.00666674*scaled_beam_draught) + (0.0571542*scaled_length_beam) + (0.0619886*scaled_froude) )
		dense2d_layer_1_output_7 = self.HyperbolicTangent( 0.0194994 + (-0.00356125*scaled_long_pos_cob) + (0.00987292*scaled_prismatic_coef) + (0.000470332*scaled_length_displacement) + (0.000336141*scaled_beam_draught) + (-0.00362375*scaled_length_beam) + (0.0087018*scaled_froude) )
		dense2d_layer_1_output_8 = self.HyperbolicTangent( -0.252904 + (-0.000564489*scaled_long_pos_cob) + (-0.0140458*scaled_prismatic_coef) + (-0.000106842*scaled_length_displacement) + (0.00883858*scaled_beam_draught) + (0.026921*scaled_length_beam) + (-0.0470564*scaled_froude) )
		dense2d_layer_1_output_9 = self.HyperbolicTangent( -0.119153 + (-0.00935629*scaled_long_pos_cob) + (-0.00899743*scaled_prismatic_coef) + (0.00036931*scaled_length_displacement) + (0.00793129*scaled_beam_draught) + (-0.00251949*scaled_length_beam) + (-0.0418439*scaled_froude) )
		dense2d_layer_1_output_10 = self.HyperbolicTangent( 0.189197 + (0.00800931*scaled_long_pos_cob) + (0.00979664*scaled_prismatic_coef) + (0.0116254*scaled_length_displacement) + (0.00166535*scaled_beam_draught) + (-0.0011839*scaled_length_beam) + (0.0615476*scaled_froude) )
		dense2d_layer_1_output_11 = self.HyperbolicTangent( -0.306247 + (-0.00604848*scaled_long_pos_cob) + (-0.0051181*scaled_prismatic_coef) + (-0.0141156*scaled_length_displacement) + (0.00966108*scaled_beam_draught) + (0.0155306*scaled_length_beam) + (-0.0357161*scaled_froude) )
		dense2d_layer_1_output_12 = self.HyperbolicTangent( 0.0262201 + (-0.00240339*scaled_long_pos_cob) + (0.00331073*scaled_prismatic_coef) + (0.00655197*scaled_length_displacement) + (0.000363926*scaled_beam_draught) + (-0.00687404*scaled_length_beam) + (0.0209637*scaled_froude) )
		dense2d_layer_1_output_13 = self.HyperbolicTangent( 0.00393279 + (0.00431933*scaled_long_pos_cob) + (-0.00757449*scaled_prismatic_coef) + (-0.00149024*scaled_length_displacement) + (-0.00140521*scaled_beam_draught) + (-0.00313172*scaled_length_beam) + (0.000911297*scaled_froude) )
		dense2d_layer_1_output_14 = self.HyperbolicTangent( -0.143463 + (0.00235137*scaled_long_pos_cob) + (-0.00949217*scaled_prismatic_coef) + (-0.00635029*scaled_length_displacement) + (0.00367302*scaled_beam_draught) + (-0.00361088*scaled_length_beam) + (-0.0417656*scaled_froude) )
		dense2d_layer_1_output_15 = self.HyperbolicTangent( -0.0524244 + (-0.000579007*scaled_long_pos_cob) + (-0.00688912*scaled_prismatic_coef) + (-0.00501321*scaled_length_displacement) + (0.00578245*scaled_beam_draught) + (-0.00115239*scaled_length_beam) + (-0.0260271*scaled_froude) )
		dense2d_layer_1_output_16 = self.HyperbolicTangent( 0.186871 + (0.0138672*scaled_long_pos_cob) + (0.0110899*scaled_prismatic_coef) + (0.012694*scaled_length_displacement) + (-0.00130399*scaled_beam_draught) + (0.00117003*scaled_length_beam) + (0.0436808*scaled_froude) )
		dense2d_layer_1_output_17 = self.HyperbolicTangent( -2.44178 + (0.0276733*scaled_long_pos_cob) + (-0.0562157*scaled_prismatic_coef) + (0.086374*scaled_length_displacement) + (-0.0954043*scaled_beam_draught) + (-0.0976248*scaled_length_beam) + (1.69049*scaled_froude) )
		dense2d_layer_1_output_18 = self.HyperbolicTangent( 0.188824 + (-0.0160566*scaled_long_pos_cob) + (0.00289666*scaled_prismatic_coef) + (0.023315*scaled_length_displacement) + (-0.0111783*scaled_beam_draught) + (0.0173967*scaled_length_beam) + (0.034998*scaled_froude) )
		dense2d_layer_1_output_19 = self.HyperbolicTangent( -0.25113 + (-0.00459456*scaled_long_pos_cob) + (-0.00524651*scaled_prismatic_coef) + (-0.0383851*scaled_length_displacement) + (-0.00213523*scaled_beam_draught) + (-0.0110634*scaled_length_beam) + (-0.0423206*scaled_froude) )
		approximation_layer_output_0 = self.Linear( 1.22237 + (-0.118989*dense2d_layer_1_output_0) + (-0.0796283*dense2d_layer_1_output_1) + (0.348712*dense2d_layer_1_output_2) + (-0.0403894*dense2d_layer_1_output_3) + (-0.12457*dense2d_layer_1_output_4) + (-0.0280108*dense2d_layer_1_output_5) + (0.356587*dense2d_layer_1_output_6) + (0.0388874*dense2d_layer_1_output_7) + (-0.278927*dense2d_layer_1_output_8) + (-0.129937*dense2d_layer_1_output_9) + (0.218112*dense2d_layer_1_output_10) + (-0.317315*dense2d_layer_1_output_11) + (0.0112616*dense2d_layer_1_output_12) + (0.00171798*dense2d_layer_1_output_13) + (-0.173803*dense2d_layer_1_output_14) + (-0.0842513*dense2d_layer_1_output_15) + (0.201162*dense2d_layer_1_output_16) + (2.37556*dense2d_layer_1_output_17) + (0.236561*dense2d_layer_1_output_18) + (-0.253934*dense2d_layer_1_output_19) )
		unscaling_layer_output_0=approximation_layer_output_0*15.16049004+10.49535751
		resistance = unscaling_layer_output_0
		outputs = [resistance]
		return outputs

	def calculate_batch_output(self, input_batch):
		output_batch = np.zeros((len(input_batch), 1))
		for i in range(len(input_batch)):
			inputs = list(input_batch[i])
			output = self.calculate_outputs(inputs)
			output_batch[i] = output
		return output_batch

def main():

	# Introduce your input values here
	long_pos_cob = 0  # long_pos_cob
	prismatic_coef = 0  # prismatic_coef
	length_displacement = 0  # length_displacement
	beam_draught = 0  # beam_draught
	length_beam = 0  # length_beam
	froude = 0  # froude

	# --- Data conversion (DO NOT modify) ---
	inputs = []

	inputs.append(long_pos_cob)
	inputs.append(prismatic_coef)
	inputs.append(length_displacement)
	inputs.append(beam_draught)
	inputs.append(length_beam)
	inputs.append(froude)

	nn = NeuralNetwork()
	outputs = nn.calculate_outputs(inputs)
	print(outputs)

if __name__ == "__main__":
	main()
