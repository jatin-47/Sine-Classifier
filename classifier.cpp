/****************************************************

Name     : Jatin Saini
Roll No. : 19ME30068

AI_2021: Neural Network Assignment Part 2

****************************************************/

//The non-linear sine classifier

#include <stdio.h>
#include <math.h>
#include <ctime>
#include <graphics.h>


//a point in 2D with the label
typedef struct
{
    int x;
    int y;
    int label;

}point;

// a structure of a node in a NN
typedef struct
{
    double wts[150];
    double b;
}node;

// initializing model design
int hid_layer1_size = 20;
int hid_layer2_size = 20;

node hid_layer1[20];
node hid_layer2[20];
node out_node;



double sigmoid(double x)
{
    return 1/(1+ exp(-x));
}

// finds the number of data points in a file
int get_size(char filename[])
{
    int size=0;
    FILE *fptr;
	fptr = fopen(filename, "r");

	char temp[20];
	//If the EOF is encountered while attempting to read a character, the EOF indicator is set (feof).
	//If this happens before any characters could be read, the pointer returned is a NULL pointer
    while(fgets(temp, 20, fptr) != NULL)
        size++;

	fclose(fptr);
	return size;
}

// reads the data file and stores the data set in an array
void read_data(int count, point data_pts[], char filename[])
{
    FILE *fptr;
	fptr = fopen(filename, "r");

	for(int i=0; i<count; i++)
	    fscanf(fptr, "%d %d %d\n", &data_pts[i].x, &data_pts[i].y, &data_pts[i].label);

    return;
}

//show all the data points on the screen
void show_points(int train_count, point train_pts[], int test_count, point test_pts[])
{
    int radius = 2;
    int color = WHITE;

    //training set
	for(int i=0; i<train_count; i++)
	{
        if(train_pts[i].label == 1)
            color = GREEN;
        else
            color = RED;
        setcolor(color);
        circle(train_pts[i].x, train_pts[i].y, radius);
        setfillstyle(SOLID_FILL, color);
		floodfill(train_pts[i].x, train_pts[i].y, color);
	}

	//test set
	color = YELLOW;
	setcolor(color);
	setfillstyle(SOLID_FILL, color);
	for(int i=0; i<test_count; i++)
	{
		circle(test_pts[i].x, test_pts[i].y, radius);
		floodfill(test_pts[i].x, test_pts[i].y, color);
    }
    return;
}

// given an input point, it gives it's label as classified by the trained model
int model_classifier(point input)
{
    double out = 0;
    double z = 0;
    double activations1[hid_layer1_size];


    for(int j=0; j<hid_layer1_size; j++)
    {
        activations1[j] = sigmoid(input.x * hid_layer1[j].wts[0] + input.y * hid_layer1[j].wts[1] + hid_layer1[j].b);
    }
    for(int j=0; j<hid_layer2_size; j++)
    {
        double temp = 0;
        for(int k=0; k<hid_layer1_size; k++)
        {
            temp += activations1[k] * hid_layer2[j].wts[k];
        }
        z += sigmoid(temp) * out_node.wts[j];
    }
    z += out_node.b;
    out = sigmoid(z);

    if(out >= 0.5)
        return 1;
    else
        return 0;
}

// trains the model
void train_model(int train_count, point train_pts[] )
{
    // initializing model with random weights and biases
    srand(time(0));
    // initializing is the key step here
    for(int i=0; i<hid_layer1_size; i++)
    {
        hid_layer1[i].wts[0] = (rand()%1001)*sqrt(2.0/2.0)/1000.0;
        hid_layer1[i].wts[1] = (rand()%1001)*sqrt(2.0/2.0)/1000.0;
        hid_layer1[i].b = 0.0;
    }
    for(int i=0; i<hid_layer2_size; i++)
    {
        for(int j=0; j<hid_layer1_size; j++)
        {
            hid_layer2[i].wts[j] = (rand()%1001)*sqrt(2.0/float(hid_layer1_size))/1000.0;
        }
        hid_layer2[i].b = 0.0;
        out_node.wts[i] = (rand()%1001)*sqrt(2.0/float(hid_layer2_size))/1000.0;
    }
    out_node.b = 0.0;

    float learnr = 0.01;
    int convergence = 0;
	int epoch_count=0;
	int pt_count=0; // counting wrongly classified points in each epoch

    double loss;
	double sq_loss;
	double sq_loss_sum;
	double delta_out;

	double activations1[hid_layer1_size];
	double activations2[hid_layer2_size];
	double z = 0;
    double out = 0;
    node grad_hid_layer1[hid_layer1_size];
    node grad_hid_layer2[hid_layer2_size];
    node grad_out_node;

    while(!convergence && epoch_count < 200)
    {
        epoch_count++;
        printf("\n\t\tEpoch %d....", epoch_count);

        pt_count = 0;
        sq_loss_sum = 0;

        // iterating through the whole training data
        for(int i=0; i<train_count; i++)
        {
            z = 0;
            for(int j=0; j<hid_layer1_size; j++)
            {
                activations1[j] = sigmoid(train_pts[i].x * hid_layer1[j].wts[0] + train_pts[i].y * hid_layer1[j].wts[1] + hid_layer1[j].b);

            }
            for(int j=0; j<hid_layer2_size; j++)
            {
                double temp = 0;
                for(int k=0; k<hid_layer1_size; k++)
                {
                    temp += activations1[k] * hid_layer2[j].wts[k];
                }
                activations2[j] = sigmoid(temp);
                z += activations2[j] * out_node.wts[j];
            }
            z += out_node.b;
            out = sigmoid(z);

            loss = (out - train_pts[i].label);
            sq_loss = loss * loss;
            sq_loss_sum += sq_loss;

            delta_out = (2 * loss) * ( out * (1-out));

            // Backpropagation
            for(int j=0; j<hid_layer2_size; j++)
            {
                grad_out_node.wts[j] = delta_out * activations2[j];

                for(int k=0; k<hid_layer1_size; k++)
                {
                    grad_hid_layer2[j].wts[k] = delta_out * out_node.wts[j] * (activations2[j] * (1 - activations2[j])) * activations1[k];
                }
                grad_hid_layer2[j].b = delta_out * out_node.wts[j] * (activations2[j] * (1 - activations2[j])) * 1;
            }
            grad_out_node.b = delta_out * 1;

            for(int j=0; j<hid_layer1_size; j++)
            {
                double sum =0;
                for(int k=0; k<hid_layer2_size; k++)
                {
                    sum += delta_out * out_node.wts[k] * (activations2[k] * (1 - activations2[k])) * hid_layer2[k].wts[j];
                }

                grad_hid_layer1[j].wts[0] = sum * (activations1[j] * (1 - activations1[j])) * train_pts[i].x;
                grad_hid_layer1[j].wts[1] = sum * (activations1[j] * (1 - activations1[j])) * train_pts[i].y;
                grad_hid_layer1[j].b = sum * (activations1[j] * (1 - activations1[j])) * 1;
            }


            // updating the parameters
            for(int i = 0; i<hid_layer1_size; i++)
            {
                hid_layer1[i].wts[0] -= learnr * grad_hid_layer1[i].wts[0];
                hid_layer1[i].wts[1] -= learnr * grad_hid_layer1[i].wts[1] ;
                hid_layer1[i].b -= learnr * grad_hid_layer1[i].b ;
            }
            for(int j=0; j<hid_layer2_size; j++)
            {
                for(int k=0; k<hid_layer1_size; k++)
                {
                    hid_layer2[j].wts[k] -= learnr * grad_hid_layer2[j].wts[k];
                }
                hid_layer2[j].b -= learnr * grad_hid_layer2[j].b;

                out_node.wts[j] -= learnr * grad_out_node.wts[j] ;
            }
            out_node.b -= learnr * grad_out_node.b;


            if((out >= 0.5 && train_pts[i].label == 0) || (out < 0.5 && train_pts[i].label == 1))
                pt_count++;
        }

        printf("\nWrongly Classified Points:\n %d/%d\n",pt_count, train_count);
        printf("Avg Loss over whole training set:\n %f\n",sq_loss_sum / train_count);


        // if MSE becomes less than 0.5, stop training
        if(sq_loss_sum / train_count < 0.001)
            convergence = 1;
    }
    printf("\n\nModel Trained Successfully!\n");
    return;
}

// test the accuracy of the classifier
void test_accuracy(int test_count, point test_pts[])
{
    float accuracy;
    int correct_count = 0;
    int radius = 2;
    int color;

    for(int i=0; i<test_count; i++)
	{
        if( model_classifier(test_pts[i]) == 1 ) //label is 1 (all upper half/on points)
        {
            color = BLUE;
            if(test_pts[i].label == 1)
                correct_count++;
        }
        else  //label is 0 (all lower half points)
        {
            color = LIGHTRED;
            if(test_pts[i].label == 0)
                correct_count++;
        }
        setcolor(color);
        circle(test_pts[i].x, test_pts[i].y, radius);
        setfillstyle(SOLID_FILL, color);
		floodfill(test_pts[i].x, test_pts[i].y, color);
		delay(5);
	}
	accuracy = (float)(correct_count*100) / test_count;
	printf("\nAccuracy of classifier:\n%f\n",accuracy);
	return;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main()
{
    //getting training and test set size
	int train_count = get_size((char *)"training_set.txt");
    int test_count = get_size((char *)"test_set.txt");

    point train_pts[train_count];
    point test_pts[test_count];

    //reading and storing training and test data
	read_data(train_count, train_pts, (char *)"training_set.txt");
	read_data(test_count, test_pts, (char *)"test_set.txt");

	int gd=DETECT, gm;
    initgraph(&gd, &gm, (char *)"");

    // showing all train and test data on screen
    show_points(train_count, train_pts, test_count, test_pts);

    // training the model
    train_model(train_count, train_pts);

    // test the accuracy of the model trained
    test_accuracy(test_count, test_pts);

    getch();
	closegraph();
	return 0;
}

