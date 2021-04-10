/****************************************************

Name     : Jatin Saini
Roll No. : 19ME30068

AI_2021: Neural Network Assignment Part 2

****************************************************/

//Generating artificial training and test data

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

// y = Amplitude*sin(frequency*x +Phase) + Offset
// So, there are 4 parameters: A,f,theta, offset
typedef struct
{
    float Amp;
    float freq;
    float phase;
    float offset;

}sin_params;


// taking parameters as input and generating points
void generate_points(int train_count, point train_pts[], int test_count, point test_pts[], sin_params sinf)
{
	srand(time(0));
	int maxx = getmaxx();
	int maxy = getmaxy();

    // we choose a random 'x' maintaining bounds so that both 'x' and 'y' lie inside the display screen
	// generating and labeling training set
	for(int i=0; i<train_count; i++)
	{
        train_pts[i].x = rand()%(maxx+1);
        train_pts[i].y = rand()%(maxy+1);

        if ( train_pts[i].y - (sinf.Amp*sin(sinf.freq*train_pts[i].x + sinf.phase) + sinf.offset) >= 0)
            train_pts[i].label = 1;          //sets the label as 1 to all upper half/on points

        else
            train_pts[i].label = 0;         //sets the label as 0 to all lower half points
	}

	// generating and labeling test set
	for(int i=0; i<test_count; i++)
	{
		test_pts[i].x = rand()%(maxx+1);
        test_pts[i].y = rand()%(maxy+1);

        if ( test_pts[i].y - (sinf.Amp*sin(sinf.freq*test_pts[i].x + sinf.phase) + sinf.offset) >= 0)
            test_pts[i].label = 1;          //sets the label as 1 to all upper half/on points

        else
            test_pts[i].label = 0;         //sets the label as 0 to all lower half points
	}
}

// stores the training set and the test set points in a .txt files
void store_points(int train_count, point train_pts[], int test_count, point test_pts[])
{
	FILE *fptr;

	fptr = fopen("training_set.txt", "w");
	for(int i=0; i<train_count; i++)
        fprintf(fptr, "%d %d %d\n", train_pts[i].x, train_pts[i].y, train_pts[i].label);
    fclose(fptr);


	fptr = fopen("test_set.txt", "w");
	for(int i=0; i<test_count; i++)
        fprintf(fptr, "%d %d %d\n", test_pts[i].x, test_pts[i].y, test_pts[i].label);
    fclose(fptr);
}

// shows the training set and test set on screen via graphics
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
void show_boundary(sin_params sinf)
{
	int radius = 1;
    int color = WHITE;
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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main()
{
    int gd=DETECT, gm;
    initgraph(&gd, &gm, (char *)"");
//    initwindow(1530, 795);

    int train_count = 1000; //training points
    int test_count = 300;  //test points

    // y = Amplitude*sin(frequency*x +Phase) + Offset
    sin_params sinf = {30, 0.05, 60, 200};

    point train_pts[train_count];
    point test_pts[test_count];

    generate_points(train_count, train_pts, test_count, test_pts, sinf);
    store_points(train_count, train_pts, test_count, test_pts);
    show_boundary(sinf);
    show_points(train_count, train_pts, test_count, test_pts);

    getch();
	closegraph();
	return 0;
}
