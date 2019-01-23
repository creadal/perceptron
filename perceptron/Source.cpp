#include <vector>
#include <cmath>
#include <iostream>
#include <set>
#include <windows.h>

using namespace std;

class Neuron
{
public:
	vector<float> input;
	int act_f;
	Neuron(int act_f)
	{
		this->act_f = act_f;
	}
	void put_in(float x)
	{
		input.push_back(x);
	}
	void put_in(vector<float> x)
	{
		input = x;
	}
	float activate()
	{
		float sum = 0;
		for (float i : input)sum += i;
		switch (act_f)
		{
		case 0: return sum;
			break;
		case 1: return 1 / (1 + exp(-sum));
			break;
		case 2: return (exp(sum) - exp(-sum)) / (exp(sum) + exp(-sum));
		}
	}
};

class Layer
{
public:
	vector<Neuron> neurons;

	Layer(int n, int act_f)
	{
		for (int i = 0; i < n; i++)
		{
			Neuron neuron(act_f);
			neurons.push_back(neuron);
		}
	}
};

class NeuralNetwork
{
public:
	vector<vector<vector<float>>> weights, individual_inputs;
	vector<Layer> layers;

	NeuralNetwork(int l, int n, int act_f)
	{
		Layer input_layer(1, 0);
		layers.push_back(input_layer);
		for (int i = 0; i < l; i++)
		{
			Layer layer(n, act_f);
			layers.push_back(layer);
		}
		Layer output_layer(1, act_f);
		layers.push_back(output_layer);

		vector<float> first;
		vector<vector<float>> second;
		for (int i = 0; i < layers[1].neurons.size() + 1; i++)first.push_back(0);
		for (int i = 0; i < layers[1].neurons.size() + 1; i++)second.push_back(first);
		for (int i = 0; i < layers.size() + 1; i++)individual_inputs.push_back(second);
	}
	void fillWeights(vector<vector<vector<float>>> weights)
	{
		this->weights = weights;
	}
	void fillWeightsWithZeros()
	{
		vector<vector<vector<float>>> weights;
		vector<float> first;
		vector<vector<float>> second;
		for (int i = 0; i < layers[1].neurons.size() + 1; i++)first.push_back(0);
		for (int i = 0; i < layers[1].neurons.size() + 1; i++)second.push_back(first);
		for (int i = 0; i < layers.size() + 1; i++)weights.push_back(second);
		this->weights = weights;
	}
	void changeWeight(int l, int n1, int n2, float value)
	{
		weights[l][n1][n2] = value;
	}
	float getResult(float input)
	{
		layers[0].neurons[0].put_in(input);
		for (int l = 0; l < layers.size() - 1; l++)
			for (int n2 = 0; n2 < layers[l + 1].neurons.size(); n2++)
			{
				for (int n1 = 0; n1 < layers[l].neurons.size(); n1++)
				{
					individual_inputs[l][n1][n2] = layers[l].neurons[n1].activate();
					layers[l + 1].neurons[n2].put_in(layers[l].neurons[n1].activate()*weights[l][n1][n2]);
				}
				layers[l + 1].neurons[n2].put_in(weights[l][weights[l].size() - 1][n2]);
			}
		float result = layers[layers.size() - 1].neurons[0].activate();
		for (int l = 0; l < layers.size(); l++) {
			for (int n = 0; n < layers[l].neurons.size(); n++) {
				layers[l].neurons[n].input.clear();
			}
		}
		return result;
	}
	void learn(float x, float y, float d, float error)
	{

	}

	float estimatedError(vector<float> inputSet, vector<float> output)
	{
		float result = 0;
		for (int i = 0; i < inputSet.size(); i++) result += abs(getResult(inputSet[i]) - output[i]);
		result /= inputSet.size();
		return result;
	}
};

void modifyRandomly(vector<vector<vector<float>>> *input, int range)
{
	vector<vector<vector<float>>> a = *input;
	for (int i = 0; i < a.size(); i++)
		for (int j = 0; j < a[i].size(); j++)
			for (int k = 0; k < a[i][j].size(); k++)
				a[i][j][k] += ((rand() % 3 - 1)*(rand() % 10 + 1)) / pow(10, range);
	*input = a;
}

void preTrain(NeuralNetwork &nn, int generations, int pop, float mut_ratio, int mut_range, vector<float> input, vector<float> output)
{
	//filling weights with random values
	for (int i = 0; i < nn.weights.size(); i++)
		for (int j = 0; j < nn.weights[i].size(); j++)
			for (int k = 0; k < nn.weights[i][j].size(); k++)
				nn.weights[i][j][k] = ((rand() % 3 - 1)*(rand() % 100 + 1) / 1);

	//creating first population
	vector<vector<vector<float>>> *population = new vector<vector<vector<float>>>[pop];
	for (int i = 0; i < 10; i++)
	{
		population[i] = nn.weights;
		modifyRandomly(&population[i], mut_range);

	}
	cout << "pretraining" << endl;
	for (int generation = 0; generation < generations; generation++)
	{
		//crossing populations
		vector<vector<vector<vector<float>>>> crossed;
		nn.fillWeightsWithZeros();
		for (int i = 0; i < pop*pop/2; i++)crossed.push_back(nn.weights);

		for (int male = 0; male < 10; male++)
			for (int female = male; female < 10; female++)
			{
				for (int i = 0; i < nn.weights.size(); i++)
					for (int j = 0; j < nn.weights[i].size(); j++)
						for (int k = 0; k < nn.weights[i][j].size(); k++)
						{
							crossed[(10 * male + female) / 2]
								[i][j][k] =
								0.5*(population[male][i][j][k] + population[female][i][j][k]);
						}
			}

		//mutating populations
		set<int> mutationIndexes;
		while (mutationIndexes.size() < pop*pop*mut_ratio /2)mutationIndexes.insert((rand() % pop*pop/2));
		for (int i : mutationIndexes) modifyRandomly(&crossed[i], mut_range);

		//selecting population
		vector<float*> errors;
		for (int i = 0; i < 50; i++)
		{
			nn.fillWeights(crossed[i]);
			float a[2] = { i, nn.estimatedError(input, output) };
			errors.push_back(a);
		}
		for (int i = 0; i < 10; i++)
		{
			int index = 0;
			for (int j = 0; j < crossed.size(); j++)
			{
				nn.fillWeights(crossed[index]);
				float min = nn.estimatedError(input, output);

				nn.fillWeights(crossed[j]);
				float current = nn.estimatedError(input, output);
				if (current < min) index = j;
			}
			population[i] = crossed[index];
			crossed.erase(crossed.begin() + index);
		}
		nn.fillWeights(population[0]);

	}
	cout << "pretraining finished" << endl;
}

void train() //backpropagation
{

}

int main()
{

	NeuralNetwork test(2, 4, 1);
	test.fillWeightsWithZeros();

	vector<float> input = { 1,2,4,8,16 }, output = { 0.1,0.2,0.4,0.8,1.6 };
	preTrain(test, 30, 10, 0.2, 1, input, output);

	while (true)
	{
		float n;
		cin >> n;
		cout << test.getResult(n) << endl;
	}

	system("pause");
}