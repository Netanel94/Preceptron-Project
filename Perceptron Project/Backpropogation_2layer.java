
public class Backpropogation_2layer {

	private int team = 3;
	private int TestUnits = training_data.test.length;
	public static int HiddenNeurons = 50;
	public static int SecondHidden = 25;
	public static int InputNeurons = 100;
	private double[] secondHiddenLayer = new double [SecondHidden];
	private double[][] WeightsSecondHidden = new double [SecondHidden][HiddenNeurons];
	private int InputLayer[] = new int[InputNeurons];
	private double HiddenLayer[] = new double[HiddenNeurons];
	private double OutputLayer;
	private double WeightsOut[] = new double[SecondHidden];
	private double WeightsHidd[][] = new double[HiddenNeurons][InputNeurons];
	private double nu=0.1;
	private double Threshold1;
	private double Threshold2;
	private boolean NetError;


	public Backpropogation_2layer() {}

	public final double returnOutput() {return OutputLayer;}
	public double learningRate() {return nu;}
	public double thresHold1() { return Threshold1;}
	public double thresHold2() { return Threshold2;}

	//_________________________________________________________________________
	
	private double randomNumber(double low,double high) {
		return (double)((Math.random() * (high-low)) + low);
	}
	
	//_________________________________________________________________________
	
	// Initializing Weights
	public void initialize() {
		Threshold1= 1.0/3;
		Threshold2= 2.0/3;
		NetError= false;

		for(int i = 0 ; i<SecondHidden; i++) {
			WeightsOut[i]= randomNumber(-1,1);
		}

		for(int i = 0;i<SecondHidden; i++) {
			for(int j = 0 ; j <HiddenNeurons; j++) {
				WeightsSecondHidden[i][j] = randomNumber(-1,1);
			}
		}
		for(int i = 0 ;i<HiddenNeurons;i++) {
			for(int j = 0; j<InputNeurons;j++) {
				WeightsHidd[i][j] =randomNumber(-1,1);
			}
		}
	}

	//_________________________________________________________________________
	
	// sigmoid fucntion
	private double sigmoid(double x) {
		double v;
		v = 1/(1+Math.exp(-x));
		return v;
	}
	
	//_________________________________________________________________________

	// Output cauculation
	private void calculateOutPut() {
		double Sum;
		for(int i = 0; i<HiddenNeurons;i++) {
			Sum=0.0;
			for(int j = 0 ;j<InputNeurons;j++) {
				Sum+= WeightsHidd[i][j]*InputLayer[j];
			}
			HiddenLayer[i] = sigmoid(Sum);
		}


		for(int i = 0;i<SecondHidden; i++) {
			Sum= 0.0;
			for(int j = 0 ; j <HiddenNeurons; j++) {
				Sum+= WeightsSecondHidden[i][j] * HiddenLayer[j];
			}
			secondHiddenLayer[i] = sigmoid(Sum);
		}

		Sum=0.0;
		for(int n = 0; n<SecondHidden;n++) {
			Sum+=WeightsOut[n]*secondHiddenLayer[n];
		}
		if(sigmoid(Sum)<Threshold1) {
			OutputLayer = 0;
		}else if (sigmoid(Sum) > Threshold1 && sigmoid(Sum) < Threshold2) {
			OutputLayer = 0.5;
		}else
			OutputLayer = 1;
	}

	//_________________________________________________________________________
	
	// Error check.
	private void isError(double target) {
		if((double)target-OutputLayer == 0.0){
			NetError = false;			
		} else { NetError = true; }
	}
	
	//_________________________________________________________________________
	
	// Weights Adjustment.
	private void adujstWeights(double target) {
		double[] hidd_deltas = new double[HiddenNeurons];
		double[] hidd_2nd_detlas = new double[SecondHidden];
		double out_delta = 0;

		// delta Weightsout
		out_delta= (sigmoid(OutputLayer)*(1-sigmoid(OutputLayer))) * (target - OutputLayer);

		// delta hidd_2 - hidden_1

		for(int i = 0; i<SecondHidden;i++) {
			hidd_2nd_detlas[i] = (sigmoid(secondHiddenLayer[i])*(1-sigmoid(secondHiddenLayer[i]))) * out_delta * WeightsOut[i];
		}


		//delta hidden1 - input 

		double[] temp = new double[HiddenNeurons]; 
		for(int i=0;i<HiddenNeurons;i++)
		{
			for(int j=0;j<SecondHidden;j++)
			{
				temp[i] += hidd_2nd_detlas[j]*WeightsSecondHidden[j][i];
			}
		}


		for(int j = 0; j <HiddenNeurons ; j++) {
			hidd_deltas[j] = (sigmoid(HiddenLayer[j])*(1-sigmoid(HiddenLayer[j]))) * temp[j];
		}


		//adjust weights

		for(int i = 0;i<SecondHidden;i++) {
			WeightsOut[i] = WeightsOut[i] + (nu * out_delta * secondHiddenLayer[i]);
		}
		for(int i = 0 ;i<SecondHidden; i++) {
			for(int j = 0 ; j<HiddenNeurons; j++) {
				WeightsSecondHidden[i][j] = WeightsSecondHidden[i][j] + (nu * hidd_2nd_detlas[i] *HiddenLayer[j]);

			}
		}
		for(int i = 0 ; i<HiddenNeurons ; i++) {
			for(int j = 0; j<InputNeurons ; j++) {
				WeightsHidd[i][j] = WeightsHidd[i][j] + (nu * hidd_deltas[i] * InputLayer[j]);

			}
		}
	}
	
	//_________________________________________________________________________
	
	// Getting a shuffled array for the random training function
	private void shuffle(int [] arr) {
		for(int i = 0 ; i<arr.length;i++) {
			int pick = (int)randomNumber(i,arr.length);
			int temp = arr[pick];
			arr[pick]=arr[i];
			arr[i] = temp;
		}

	}
	
	//_________________________________________________________________________
	
	// Random Training
	public boolean trainNetRandom(int Units)
	{
		int Matricis = Units*3;
		int Error;
		int loop = 0;
		int Success;

		do {
			Error = 0;
			loop ++;
			for(int j = 0; j<Units ;j++) {

				int unitsIndex[] = new int[team];
				for(int index = 0 ; index<team ; index++) {
					unitsIndex[index]=index;
				}
				shuffle(unitsIndex);
				for(int i=0;i<unitsIndex.length;i++) {
					for(int r=0;r<10;r++)
						for(int c=0;c<10;c++)
							InputLayer[10*r+c] = training_data1.inputs[j][unitsIndex[i]][r][c] == '*' ? 1:-1;

					calculateOutPut();
					isError(training_data1.outputs[unitsIndex[i]]);

					if(NetError) {
						Error++;
						adujstWeights(training_data1.outputs[unitsIndex[i]]);
					}
				}

			}

			Success=((Matricis-Error)*100)/Matricis;
			
			if(loop == 1) {
				System.out.println("Succsess: " + Success+"%"  + " loop: " + loop);
			}
			if(loop % 10 == 0) {
				System.out.println("Succsess: " + Success+"%"  + " loop: " + loop);
			}
			if(Success >= 80) {
				System.out.println("Succsess: " + Success+"%"  + " loop: " + loop);
			}



		}while(Success < 80 && loop<=10000);


		if(loop > 10000) {
			return false;
		}else
			return true;		
	}
	
	//_________________________________________________________________________
	
	// Normal Training
	public boolean trainNet(int Units)
	{
		int Matricis = Units*3;
		int Error;
		int loop = 0;
		int Success;

		do {
			Error = 0;
			loop ++;
			
			for(int j = 0; j<Units ;j++) 
			{
				for(int i=0;i<team;i++) 
				{
					for(int r=0;r<10;r++)
						for(int c=0;c<10;c++)
							InputLayer[10*r+c] = training_data1.inputs[j][i][r][c] == '*' ? 1:-1;
					calculateOutPut();
					isError(training_data1.outputs[i]);

					if(NetError) 
					{
						Error++;
						adujstWeights(training_data1.outputs[i]);
					}
				}

			}

			Success=((Matricis-Error)*100)/Matricis;
			
			if(loop == 1) {
				System.out.println("Succsess: " + Success+"%"  + " loop: " + loop);
			}
			if(loop % 10 == 0) {
				System.out.println("Succsess: " + Success+"%"  + " loop: " + loop);
			}
			if(Success >= 80) {
				System.out.println("Succsess: " + Success+"%"  + " loop: " + loop);
			}


		}while(Success < 80 && loop<=10000);


		if(loop > 10000) {
			return false;
		}else
			return true;		
	}
	
	//_________________________________________________________________________
	
	//Normal Test
	public int testNet() {
		int Error = 0;
		int Success;

		for(int i=0;i<training_data1.test.length;i++)
		{
			for(int r=0;r<10;r++)
				for(int c=0;c<10;c++)
					InputLayer[10*r+c] = training_data1.test[i][r][c] == '*' ? 1:-1;

			calculateOutPut();
			isError(training_data1.outputs[i]);

			if(NetError) {
				Error++;
			}
		}
		Success=((training_data1.test.length-Error)*100)/training_data1.test.length;
		
		return Success;
	}
	
	// Random Test
	public int testNetRandom() {
		int Error = 0;
		int Success;
		int unitsIndex[] = new int[TestUnits];
		for(int index = 0 ; index<TestUnits ; index++) {
			unitsIndex[index]=index;
		}
		shuffle(unitsIndex);
		for(int i=0;i<unitsIndex.length;i++) {
			for(int r=0;r<10;r++)
				for(int c=0;c<10;c++)
					InputLayer[10*r+c] = training_data1.test[unitsIndex[i]][r][c] == '*' ? 1:-1;

			calculateOutPut();
			isError(training_data1.outputs[unitsIndex[i]]);

			if(NetError) {
				Error++;
			}
		}
		Success=((training_data1.test.length-Error)*100)/training_data1.test.length;

		return Success;
	}

}





