import java.util.Scanner;
public class Tests {

	// This is the Main function for both Backpropogation classes.

	public static void main(String[] args) {

		// Driver for 1 Hidden Layer
		Scanner scan = new Scanner(System.in);
		System.out.println("Please Enter only The number of groups you want to train between 1 - 20 :");
		int k = scan.nextInt();
		System.out.println("-----------------------------");
		System.out.println(" This is Backproogation progarm for both 1 Hidden Layer and 2 Hidden Layers");
		Backpropogation1 back = new Backpropogation1();
		back.initialize();
		System.out.println("-----------------------------");
		System.out.println("This is Normal Training Sequence with 1 Hidden Layer");
		System.out.println("-----------------------------");
		if(back.trainNet(k)) {								//  <------ Number of Groups
			System.out.println("Training was Successful");
		}else
			System.out.println("Training of netwrok was a failure!");	
		System.out.println("-------------------------");
		System.out.print("This is a normal test for 1 Hidden Layer: ");
		System.out.println(back.testNet() + "%");
		System.out.println("---------------------------");
		System.out.println("                           ");
		System.out.println("-----------------------------");
		System.out.print("This is a Random Order test for 1 Hidden Layer after a normal Training session: ");
		System.out.println(back.testNetRandom() + "%");
		System.out.println("-----------------------------");

		System.out.println("                           ");
		System.out.println("This is Random Order training Sequence with 1 Hidden Layer");
		System.out.println("-----------------------------");
		System.out.println("                           ");
		back.initialize();
		if(back.trainNetRandom(k)) {						 //  <------ Number of Groups 
			System.out.println("                           ");
			System.out.println("Training was Successful");
		}else
			System.out.println("Training of network was a failure!");

		System.out.println("                           ");
		System.out.println("-----------------------------");
		System.out.print("This is a Random Order test for 1 Hidden Layer: ");
		System.out.println(back.testNetRandom() + "%");
		System.out.println("-----------------------------");
		System.out.println("-------------------------");
		System.out.print("This is a normal test for 1 Hidden Layer for a Random Order Training session: ");
		System.out.println(back.testNet() + "%");
		System.out.println("---------------------------");

		// ----------------------------------
		// Driver for 2 Hidden Layers
		// ----------------------------------
		System.out.println("                           ");
		System.out.println("                           ");

		Backpropogation_2layer back2 = new Backpropogation_2layer();
		back2.initialize();
		System.out.println("-----------------------------");
		System.out.println("This is Normal Training Sequence with 2 Hidden Layers");
		System.out.println("-----------------------------");
		if(back2.trainNet(k)) {								//  <------ Number of Groups
			System.out.println("Training was Successful");
		}else
			System.out.println("Training of network was a failure!");
		System.out.println("-------------------------");
		System.out.print("This is a normal test for 2 Hidden Layers: ");
		System.out.println(back2.testNet() + "%");
		System.out.println("---------------------------");
		System.out.print("This is a Random Order test for 2 Hidden Layers for a Normal Training Session: ");
		System.out.println(back2.testNetRandom() + "%");
		System.out.println("-----------------------------");
		System.out.println("                           ");
		System.out.println("This is Random Order training Sequence with 2 Hidden Layers");
		System.out.println("-----------------------------");
		back2.initialize();
		if(back2.trainNetRandom(k)) {						 //  <------  Number of Groups 
			System.out.println("                           ");
			System.out.println("Training was Successful");
		}else
			System.out.println("Training of network was a failure!");

		System.out.println("                           ");
		System.out.println("-----------------------------");
		System.out.print("This is a Random Order test for 2 Hidden Layers: ");
		System.out.println(back2.testNetRandom() + "%");
		System.out.println("-----------------------------");
		System.out.print("This is a normal test for 2 Hidden Layers for a Random Order Training Session: ");
		System.out.println(back2.testNet() + "%");
		System.out.println("---------------------------");
		System.out.println("                           ");
	}
}





