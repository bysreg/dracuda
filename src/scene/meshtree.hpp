//
//  MeshTree.h
//  p4
//
//  Created by Nathan Dobson on 11/11/14.
//  Copyright (c) 2014 Nathan Dobson. All rights reserved.
//

#ifndef __p4__MeshTree__
#define __p4__MeshTree__

#include <stdio.h>
#include <scene/scene.hpp>

#include "scene/mesh.hpp"
#include "scene/bound.hpp"

namespace _462 {

#define MAX_TREE_DEPTH 50

class ModelBody;

class MeshTreeNode{
public:
	MeshTreeNode();
	MeshTreeNode(const Mesh* mesh, std::vector<MeshTriangle> triangles, Matrix4 mat);
	~MeshTreeNode();

	Bound* bbox;
	MeshTreeNode* left;
	MeshTreeNode* right;
	
	std::vector<MeshTriangle> triangles;

	// get a bounding box surrounding all the triangles
	Bound* createBBox(const Mesh* mesh, Matrix4 mat);
	// get a bounding box from triangle
	Bound createTriangleBound(const Mesh* mesh, MeshTriangle triangle, Matrix4 mat);
};

class MeshTree{
public:
	MeshTreeNode* root;

	MeshTree();
	MeshTree(const Mesh *mesh, Matrix4 mat);
	~MeshTree();

	MeshTree *makeFlatTree(const Mesh *mesh);
	void refine(const Mesh *mesh, MeshTree *tree);
	MeshTree *makeRecTree(const Mesh *mesh);
	void refineRec(const Mesh *mesh, MeshTree *tree);
	Bound setBounds(const Mesh *mesh, MeshTree *tree);

	void freeNode(MeshTreeNode* node);
	MeshTreeNode* build(const Mesh *mesh, std::vector<MeshTriangle> triangles, Matrix4 mat, int depth);
	Vector3 getTriangleMidPoint(const Mesh *mesh, MeshTriangle triangle);
};
} //namespace _462
#endif /* defined(__p4__MeshTree__) */
