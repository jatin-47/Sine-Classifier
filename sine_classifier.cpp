/****************************************************

Name     : Jatin Saini
Roll No. : 19ME30068

AI_2021: Neural Network Assignment Part 2

****************************************************/

//The non-linear sine classifier

#include <stdio.h>
#include <math.h>
#include <graphics.h>

//a point in 2D with the label
typedef struct
{
    int x;
    int y;
    int label;

}point;

// y = Amplitude*sin(frequency*x +Phase) + Offset
// So, there are 4 parameters: A,f,theta, offset
typedef struct
{
    float Amp;
    float freq;
    float phase;
    float offset;

}sin_params;

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

// draw the graph of a sine function using the passed parameters
void show_boundary(sin_params sinf, int color)
{
	int radius = 1;
	setcolor(color);
	setfillstyle(SOLID_FILL, color);

    int maxx = getmaxx();
	point boundary[maxx+1];

	for(int i=0; i<=maxx; i++)
    {
        boundary[i].x = i;
        boundary[i].y = sinf.Amp*sin(sinf.freq*boundary[i].x + sinf.phase) + sinf.offset;

        circle(boundary[i].x, boundary[i].y, radius);
		floodfill(boundary[i].x, boundary[i].y, color);
    }
    return;
}

// test the accuracy of the classifier
void test_accuracy(int test_count, point test_pts[], sin_params w)
{
    float accuracy;
    int correct_count = 0;
    int radius = 2;
    int color;

    for(int i=0; i<test_count; i++)
	{
        if( (test_pts[i].y - (w.Amp*sin(w.freq*test_pts[i].x + w.phase) + w.offset)) >= 0 ) //label is 1 (all upper half/on points)
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

// learning algorithm
sin_params grad_dec(int train_count, point train_pts[])
{
    // {50, 0.05, 40, 250}
    sin_params w = {0, 0.05, 0, 0};
	float learnr = 40;
    show_boundary(w, WHITE);

	double e;
	double output;
	double loss;
	double sq_loss;
	double temp;

	double sum_sq_loss = 0;
	double sum_dev_w[4] ={0};

	int convergence = 0;
	int epoch_count=0;

	while(!convergence && epoch_count < 100000)
    {
        epoch_count++;
        printf("\n\t\tEpoch %d....", epoch_count);

        sum_sq_loss = 0;
        sum_dev_w[0] = 0; sum_dev_w[1] = 0; sum_dev_w[2] = 0; sum_dev_w[3] = 0;
        int pt_count=0;

        for(int i=0; i<train_count; i++)
        {
            e = exp(-(train_pts[i].y - (w.Amp*sin(w.freq*train_pts[i].x + w.phase) + w.offset)));
            output = 1 /(1+e);
            loss = train_pts[i].label - output;
            sq_loss = loss*loss;
            sum_sq_loss += sq_loss;
            if( sq_loss > 0.25 )
            {
                pt_count++;
                temp = (2 * loss * e) / ((1+e)*(1+e));

                sum_dev_w[0] += temp * sin(w.freq*train_pts[i].x + w.phase);
               // sum_dev_w[1] += temp * w.Amp * cos(w.freq*train_pts[i].x + w.phase)* train_pts[i].x;
                sum_dev_w[2] += temp * w.Amp * cos(w.freq*train_pts[i].x + w.phase);
                sum_dev_w[3] += temp;
            }
        }
        printf("\nWrong Points:\n %d\n",pt_count);
        if(pt_count != 0)
        {
            sum_sq_loss /= train_count;
            sum_dev_w[0] /= pt_count;
            sum_dev_w[1] /= pt_count;
            sum_dev_w[2] /= pt_count;
            sum_dev_w[3] /= pt_count;
        }
        printf("Avg Loss over all wrong classified points:\n %f\n",sum_sq_loss);

        w.Amp -= learnr * sum_dev_w[0];
//        w.freq -= learnr * sum_dev_w[1];
        w.phase -= learnr * sum_dev_w[2];
        w.offset -= learnr * sum_dev_w[3];

        printf("Prediction(Amp,Freq,Phase,Offset):\n %f %f %f %f \n", w.Amp, w.freq, w.phase, w.offset);

        if(epoch_count%10000 == 0)
            show_boundary(w, WHITE);
        if(epoch_count == 10000)
            learnr = 0.1;

        int i;
        for(i=0; i<train_count; i++)
        {
            if( ((train_pts[i].y - (w.Amp*sin(w.freq*train_pts[i].x + w.phase) + w.offset)) >= 0 && train_pts[i].label == 0) ||
                ((train_pts[i].y - (w.Amp*sin(w.freq*train_pts[i].x + w.phase) + w.offset)) < 0 && train_pts[i].label == 1)       )
            {
                convergence = 0;
                break;
            }
        }
        if(i == train_count)
            convergence = 1;
    }
    printf("\n\n%f %f %f %f \n PREDICTED!", w.Amp, w.freq, w.phase, w.offset);
    return w;
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

    show_points(train_count, train_pts, test_count, test_pts);

    sin_params w = grad_dec(train_count, train_pts);

    cleardevice();
    show_boundary(w, WHITE);
	show_points(train_count, train_pts, test_count, test_pts);
    test_accuracy(test_count, test_pts, w);

    getch();
	closegraph();
	return 0;
}
