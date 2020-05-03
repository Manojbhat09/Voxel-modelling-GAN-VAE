/* ////////////////////////////////////////////////////////////

File Name: main.cpp
Copyright (c) 2017 Soji Yamakawa.  All rights reserved.
http://www.ysflight.com

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//////////////////////////////////////////////////////////// */

#include <fslazywindow.h>
#include <ysglfontdata.h>
#include <stdio.h>
#include <math.h>
#include <vector>

using namespace std;
const double YsPi = 3.1415927;


class CameraObject
{
public:
	double x, y, z;
	double h, p, b;

	double fov, nearZ, farZ;

	CameraObject();
	void Initialize(void);
	void SetUpCameraProjection(void);
	void SetUpCameraTransformation(void);

	void GetForwardVector(double& vx, double& vy, double& vz);
};

CameraObject::CameraObject()
{
	Initialize();
}

void CameraObject::Initialize(void)
{
	x = 0;
	y = 0;
	z = 0;
	h = 0;
	p = 0;
	b = 0;

	fov = YsPi / 6.0;  // 30 degree
	nearZ = 0.1;
	farZ = 200.0;
}

void CameraObject::SetUpCameraProjection(void)
{
	int wid, hei;
	double aspect;

	FsGetWindowSize(wid, hei);
	aspect = (double)wid / (double)hei;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fov * 180.0 / YsPi, aspect, nearZ, farZ);
}

void CameraObject::SetUpCameraTransformation(void)
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glRotated(-b * 180.0 / YsPi, 0.0, 0.0, 1.0);  // Rotation about Z axis
	glRotated(-p * 180.0 / YsPi, 1.0, 0.0, 0.0);  // Rotation about X axis
	glRotated(-h * 180.0 / YsPi, 0.0, 1.0, 0.0);  // Rotation about Y axis
	glTranslated(-x, -y, -z);
}

void CameraObject::GetForwardVector(double& vx, double& vy, double& vz)
{
	vx = -cos(p) * sin(h);
	vy = sin(p);
	vz = -cos(p) * cos(h);
}


class OrbitingViewer
{
public:
	double h, p;
	double dist;
	double focusX, focusY, focusZ;

	OrbitingViewer();
	void Initialize(void);
	void SetUpCamera(CameraObject& camera);
};

OrbitingViewer::OrbitingViewer()
{
	Initialize();
}

void OrbitingViewer::Initialize(void)
{
	h = 0;
	p = 0;
	dist = 100.0;
	focusX = 0.0;
	focusY = 0.0;
	focusZ = 0.0;
}

void OrbitingViewer::SetUpCamera(CameraObject& camera)
{
	camera.h = h;
	camera.p = p;
	camera.b = 0.0;

	double vx, vy, vz;
	camera.GetForwardVector(vx, vy, vz);
	camera.x = focusX - vx * dist;
	camera.y = focusY - vy * dist;
	camera.z = focusZ - vz * dist;
}

void DrawCube(double x1, double y1, double z1, double x2, double y2, double z2, int color)
{
	if (color == 0)
	{
		glColor3ub(255, 0, 0);
	}
	else if (color == 1)
	{
		glColor3ub(0, 255, 0);
	}
	else if (color == 2)
	{
		glColor3ub(0, 0, 255);
	}
	else if (color == 3)
	{
		glColor3ub(255, 255, 0);
	}
	else if (color == 4)
	{
		glColor3ub(0, 255, 255);
	}
	glBegin(GL_QUADS);

	glVertex3d(x1, y1, z1);
	glVertex3d(x2, y1, z1);
	glVertex3d(x2, y2, z1);
	glVertex3d(x1, y2, z1);

	glVertex3d(x1, y1, z2);
	glVertex3d(x2, y1, z2);
	glVertex3d(x2, y2, z2);
	glVertex3d(x1, y2, z2);

	glVertex3d(x1, y1, z1);
	glVertex3d(x2, y1, z1);
	glVertex3d(x2, y1, z2);
	glVertex3d(x1, y1, z2);

	glVertex3d(x1, y2, z1);
	glVertex3d(x2, y2, z1);
	glVertex3d(x2, y2, z2);
	glVertex3d(x1, y2, z2);

	glVertex3d(x1, y1, z1);
	glVertex3d(x1, y2, z1);
	glVertex3d(x1, y2, z2);
	glVertex3d(x1, y1, z2);

	glVertex3d(x2, y1, z1);
	glVertex3d(x2, y2, z1);
	glVertex3d(x2, y2, z2);
	glVertex3d(x2, y1, z2);

	glEnd();

	glColor3ub(255, 255, 255);

	glBegin(GL_LINES);
	glVertex3d(x1, y1, z1);
	glVertex3d(x1, y1, z2);
	glVertex3d(x1, y1, z2);
	glVertex3d(x2, y1, z2);
	glVertex3d(x2, y1, z2);
	glVertex3d(x2, y1, z1);
	glVertex3d(x2, y1, z1);
	glVertex3d(x1, y1, z1);
	glVertex3d(x1, y2, z1);
	glVertex3d(x1, y2, z2);
	glVertex3d(x1, y2, z2);
	glVertex3d(x2, y2, z2);
	glVertex3d(x2, y2, z2);
	glVertex3d(x2, y2, z1);
	glVertex3d(x2, y2, z1);
	glVertex3d(x1, y2, z1);
	glVertex3d(x1, y1, z1);
	glVertex3d(x1, y2, z1);
	glVertex3d(x1, y1, z2);
	glVertex3d(x1, y2, z2);
	glVertex3d(x2, y1, z2);
	glVertex3d(x2, y2, z2);
	glVertex3d(x2, y1, z1);
	glVertex3d(x2, y2, z1);
	glEnd();
}

void LoopingInput(vector<vector<vector<float>>> input)
{
	int initialCubeSize = 1;
	int adjust = 16;
	for (int i = 0; i < input.size(); i++)
	{
		for (int j = 0; j < input[i].size(); j++)
		{
			for (int k = 0; k < input[i][j].size(); k++)
			{
				if (input[i][j][k] > 0.5f)
				{
					DrawCube(i - adjust, j- adjust, k- adjust,
						     i - adjust + initialCubeSize, j - adjust + initialCubeSize, k - adjust + initialCubeSize, 0);
				}
				else
				{
					continue;
				}
			}
		}
	}
}

void LoopingInputWithXYZ(vector<vector<vector<float>>> input, float xPos, float yPos, float zPos, float outCubeSize, int color)
{
	//int outCubeSize = 1;
	int adjust = 16;
	for (int i = 0; i < input.size(); i++)
	{
		for (int j = 0; j < input[i].size(); j++)
		{
			for (int k = 0; k < input[i][j].size(); k++)
			{
				if (input[i][j][k] > 0.5f)
				{
					DrawCube(i - adjust + xPos, j - adjust + yPos, k - adjust + zPos,
						i - adjust + outCubeSize + xPos, j - adjust + outCubeSize + yPos, k - adjust + outCubeSize + zPos, color);
				}
				else
				{
					continue;
				}
			}
		}
	}
}

void drawText(char str[256], int width, int height, int fontSize)
{
	glRasterPos2d(width, height);
	if (fontSize == 0)
	{
		YsGlDrawFontBitmap12x16(str);
		
	}
	else if (fontSize == 1)
	{
		YsGlDrawFontBitmap16x20(str);
	}
	else if (fontSize == 2)
	{
		YsGlDrawFontBitmap20x28(str);
	}
}


void arrangeText()
{
	glRasterPos2d(300, 300);
	char str1[256];
	sprintf(str1, "Voxel Visualizer");
	YsGlDrawFontBitmap20x32(str1);
}
/*testPart*/
vector<vector<vector<float>>> generateTestVector(vector<vector<vector<float>>> vec)
{
	

	for (int i = 0; i < 32; i++)
	{
		vec.push_back(vector<vector<float>>());
		for (int j = 0; j < 32; j++)
		{
			vec[i].push_back(vector<float>());
			for (int k = 0; k < 32; k++)
			{
				if (i == j == k)
				{
					vec[i][j].push_back(1.0f);
				}
				else
				{
					vec[i][j].push_back(0.0f);
				}
			}
		}
	}

	return vec;
}

class FsLazyWindowApplication : public FsLazyWindowApplicationBase
{
protected:
	bool needRedraw;
	CameraObject camera;
	OrbitingViewer orbit;
	int wid = 800;
	int hei = 600;

	/* test part*/
	vector<vector<vector<float>>> testVec;

public:
	FsLazyWindowApplication();
	virtual void BeforeEverything(int argc, char* argv[]);
	virtual void GetOpenWindowOption(FsOpenWindowOption& OPT) const;
	virtual void Initialize(int argc, char* argv[]);
	virtual void Interval(void);
	virtual void BeforeTerminate(void);
	virtual void Draw(void);
	virtual bool UserWantToCloseProgram(void);
	virtual bool MustTerminate(void) const;
	virtual long long int GetMinimumSleepPerInterval(void) const;
	virtual bool NeedRedraw(void) const;
};

FsLazyWindowApplication::FsLazyWindowApplication()
{
	needRedraw = false;
}

/* virtual */ void FsLazyWindowApplication::BeforeEverything(int argc, char* argv[])
{
}
/* virtual */ void FsLazyWindowApplication::GetOpenWindowOption(FsOpenWindowOption& opt) const
{
	opt.x0 = 0;
	opt.y0 = 0;
	opt.wid = wid;
	opt.hei = hei;
}
/* virtual */ void FsLazyWindowApplication::Initialize(int argc, char* argv[])
{
	camera.z = 10.0;
}
/* virtual */ void FsLazyWindowApplication::Interval(void)
{
	auto key = FsInkey();
	if (FSKEY_ESC == key)
	{
		SetMustTerminate(true);
	}



	if (0 != FsGetKeyState(FSKEY_LEFT))
	{
		orbit.h += YsPi / 180.0;
	}
	if (0 != FsGetKeyState(FSKEY_RIGHT))
	{
		orbit.h -= YsPi / 180.0;
	}
	if (0 != FsGetKeyState(FSKEY_UP))
	{
		orbit.p += YsPi / 180.0;
	}
	if (0 != FsGetKeyState(FSKEY_DOWN))
	{
		orbit.p -= YsPi / 180.0;
	}
	if (0 != FsGetKeyState(FSKEY_F) && orbit.dist > 1.0)
	{
		orbit.dist /= 1.02;
	}
	if (0 != FsGetKeyState(FSKEY_B) && orbit.dist < 200.0)
	{
		orbit.dist *= 1.02;
	}
	else
	{
		orbit.h -= (YsPi / 180.0)*0.1f;
	}
	orbit.SetUpCamera(camera);



	needRedraw = true;
}
/* virtual */ void FsLazyWindowApplication::Draw(void)
{
	needRedraw = false;

	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	glViewport(0, 0, wid, hei);

	// Set up 3D drawing
	camera.SetUpCameraProjection();
	camera.SetUpCameraTransformation();

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(1, 1);

	// 3D drawing from here
	//DrawCube(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0);

	/*test part*/
	//LoopingInput(generateTestVector(testVec));

	LoopingInputWithXYZ(generateTestVector(testVec), -20, 0, 0, 1, 0);
	LoopingInputWithXYZ(generateTestVector(testVec), 20, 0, 0, 2, 1);

	arrangeText();

	// Set up 2D drawing
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, (float)wid - 1, (float)hei - 1, 0, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glDisable(GL_DEPTH_TEST);

	FsSwapBuffers();
}
/* virtual */ bool FsLazyWindowApplication::UserWantToCloseProgram(void)
{
	return true; // Returning true will just close the program.
}
/* virtual */ bool FsLazyWindowApplication::MustTerminate(void) const
{
	return FsLazyWindowApplicationBase::MustTerminate();
}
/* virtual */ long long int FsLazyWindowApplication::GetMinimumSleepPerInterval(void) const
{
	return 10;
}
/* virtual */ void FsLazyWindowApplication::BeforeTerminate(void)
{
}
/* virtual */ bool FsLazyWindowApplication::NeedRedraw(void) const
{
	return needRedraw;
}


static FsLazyWindowApplication* appPtr = nullptr;

/* static */ FsLazyWindowApplicationBase* FsLazyWindowApplicationBase::GetApplication(void)
{
	if (nullptr == appPtr)
	{
		appPtr = new FsLazyWindowApplication;
	}
	return appPtr;
}
