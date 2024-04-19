#%%
import factorySim.baseConfigs as baseConfigs
from factorySim.factorySimClass import FactorySim
from tqdm.auto import tqdm
import os
import ifcopenshell
from ifcopenshell.api import run
from ifcopenshell.util.shape_builder import ShapeBuilder
from mathutils import Vector
from shapely.geometry import box, MultiPoint, Polygon, MultiPolygon
from shapely.affinity import  scale

import math
import numpy as np

factoryConfig = baseConfigs.SMALLSQUARE



if __name__ == "__main__":

    print("Creating Factory")
    basePath = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    outputPath = os.path.join(basePath, "Output")
    #Create directory if it does not exist
    ifcPath = os.path.join(basePath, "..","Input", "2", "Basic.ifc")
    print(ifcPath)

    factory = FactorySim(path_to_ifc_file=ifcPath,factoryConfig=factoryConfig, createMachines=True, randomPos=False)
    print(factory.machine_dict)





# %%

def prepare_for_export(element_dict, bb):
    new_dict = element_dict.copy()
    for element in new_dict.values():
        element.poly = scale(element.poly, yfact=-1, origin=bb.centroid)
        polybbox = element.poly.bounds
        element.origin = (polybbox[0], polybbox[1])


    return new_dict
def write_ifc_class(ifc_file, ifc_context, ifc_class, element_dict):
    builder = ShapeBuilder(ifc_file)
    elements = []

    for element in element_dict.values():
        print(element.name)

        ifc_element = run("root.create_entity", ifc_file, ifc_class=ifc_class, name=element.name)
        #matrix[:,3][0:3] = (element.origin[0]/1000, element.origin[1]/1000, 0)

        run("geometry.edit_object_placement", model, product=ifc_element)
        #run("geometry.edit_object_placement", ifc_file, product=ifc_element, matrix=matrix, is_si=True)

        breps = []
        for poly in element.poly.geoms:

            # Create IfcCartesianPoints for outer polygon
            outer_polygon_cartesian_points = [
                model.createIfcCartesianPoint(Coordinates=(*point, 0.0)) for point in poly.exterior.coords[:-1]
                
            ]

            outer_polygon_loop = model.createIfcPolyLoop(Polygon=outer_polygon_cartesian_points)

            # Create IfcFaceBound for outer polygon
            outer_polygon_face_bound = model.createIfcFaceOuterBound(Bound=outer_polygon_loop, Orientation=False)

            faces = [outer_polygon_face_bound]

            # Create IfcFaceBound and IfcFaceOuterBound for each hole and add them to the outer face
            for ring in poly.interiors:
                # Create IfcCartesianPoints for inner hole
                inner_hole_cartesian_points = [
                    model.createIfcCartesianPoint(Coordinates=point) for point in ring.coords[:-1]
                ]

                # Create IfcPolyLoop for inner hole
                inner_hole_loop = ifcopenshell.create_entity(
                    "IfcPolyLoop",
                    Polygon=inner_hole_cartesian_points
                )

                # Create IfcFaceBound for inner hole
                inner_hole_face_bound = ifcopenshell.create_entity(
                    "IfcFaceBound",
                    Bound=inner_hole_loop,
                    Orientation=False
                )

                # Add the inner hole to the outer face
                faces.append(inner_hole_face_bound)

            # Create IfcFace for outer polygon
            outer_polygon_face = model.createIfcFace(Bounds=faces)

            # Create IfcClosedShell
            closed_shell = model.createIfcClosedShell(CfsFaces=[outer_polygon_face])

            # Create IfcFacetedBrep
            faceted_brep = model.createIfcFacetedBrep(Outer=closed_shell)
            breps.append(faceted_brep)




        representation = model.createIfcShapeRepresentation(ContextOfItems=ifc_context, RepresentationIdentifier="Body", RepresentationType="Brep", Items=breps)
        run("geometry.assign_representation", ifc_file, product=ifc_element, representation=representation)
        elements.append(ifc_element)
    return elements

# Create a blank model
model = ifcopenshell.file()

# All projects must have one IFC Project element
project = run("root.create_entity", model, ifc_class="IfcProject", name="FactorySim Project")

# Geometry is optional in IFC, but because we want to use geometry in this example, let's define units
# Assigning without arguments defaults to metric units
length = run("unit.add_si_unit", model, unit_type="LENGTHUNIT", prefix="MILLI")
run("unit.assign_unit", model, units=[length])

# Let's create a modeling geometry context, so we can store 3D geometry (note: IFC supports 2D too!)
model3d = run("context.add_context", model, context_type="Model")

plan  = run("context.add_context", model, context_type="Plan")

# In particular, in this example we want to store the 3D "body" geometry of objects, i.e. the body shape
body = run("context.add_context", model, context_type="Model",
    context_identifier="Body", target_view="PLAN_VIEW", parent=model3d)



# Create a site, building, and storey. Many hierarchies are possible.
site = run("root.create_entity", model, ifc_class="IfcSite", name="IfcSite")
building = run("root.create_entity", model, ifc_class="IfcBuilding", name="IfcBuilding")
storey = run("root.create_entity", model, ifc_class="IfcBuildingStorey", name="IfcBuildingStorey")

# Since the site is our top level location, assign it to the project
# Then place our building on the site, and our storey in the building
run("aggregate.assign_object", model, relating_object=project, product=site)
run("aggregate.assign_object", model, relating_object=site, product=building)
run("aggregate.assign_object", model, relating_object=building, product=storey)

export= prepare_for_export(factory.machine_dict, factory.factoryCreator.bb)
elements = write_ifc_class(model, body, "IfcBuildingElementProxy", export)
run("spatial.assign_container", model, relating_structure=storey, products=elements)

export= prepare_for_export(factory.wall_dict, factory.factoryCreator.bb)
elements = write_ifc_class(model, body, "IfcWall", export)
run("spatial.assign_container", model, relating_structure=storey, products=elements)


# Write out to a file
model.write("model.ifc")






# %%
ifcPath = [os.path.join(basePath, "..","Input", "2", "Basic.ifc"), "model.ifc"]
elementName = "IFCBUILDINGELEMENTPROXY"
for ifc in ifcPath:
    
    model = ifcopenshell.open(ifc)
    element = model.by_type(elementName)[5]
    print(element.Name)
    print(element.Representation.Representations[0].RepresentationIdentifier)
    print(element.Representation.Representations[0].RepresentationType)
    print(element.Representation.Representations[0].Items)
    print(element.Representation.Representations[0].Items[0].Outer)
    print(dir(element.Representation.Representations[0]))

settings = ifcopenshell.geom.settings()
shape = ifcopenshell.geom.create_shape(settings, element)
    # print(shape.geometry)
    # print(shape.context)
    # print(shape.product)
    # dir(shape)

# %%
print(ifcopenshell.util.shape.get_shape_matrix(shape))
#
# %%
element.ObjectPlacement.RelativePlacement.Location.Coordinates
# %%
x = element.ObjectPlacement.RelativePlacement.RefDirection.DirectionRatios[0]
y = element.ObjectPlacement.RelativePlacement.RefDirection.DirectionRatios[1]
rotation = math.atan2(y,x)
print(x,y,rotation)
matrix= (ifcopenshell.util.shape.get_shape_matrix(shape))
print(matrix[0,0], matrix[1,0])
# %%
print(shape.geometry.faces)
print(ifcopenshell.util.shape.get_vertices(shape.geometry))
print(ifcopenshell.util.shape.get_shape_vertices(shape, shape.geometry))
print(ifcopenshell.util.shape.get_edges(shape.geometry))
poly = Polygon(ifcopenshell.util.shape.get_shape_vertices(shape, shape.geometry)[0:3])
# %%
items = element.Representation.Representations[0].Items
for item in items:
    faces = item.Outer.CfsFaces
    for face in faces:
        exterior = []
        interior = []
        bounds = face.Bounds
        for bound in bounds:
            print(bound.get_info()['type'])
            points = bound.Bound.Polygon
            pointlist = [(point.Coordinates[0], point.Coordinates[1]) for point in points ]
            print(pointlist)
# %%
poly
# %%
print(shape.geometry.edges)
print(shape.geometry.verts)

# %%
ifcopenshell.util.unit.calculate_unit_scale(model)

# %%
first_poly= list(factory.machine_dict.values())[0]
vertices = [list(poly.exterior.coords) for poly in first_poly.poly.geoms]

vertices = []

for poly in first_poly.poly.geoms:
    lineString = [tuple([point[0], point[1], 0]) for point in poly.exterior.coords]
    vertices.append(lineString)

print(vertices)

edges = []
num_vertices = len(vertices)
print(num_vertices)
for i in range(num_vertices - 1):
    print(i)
    edges.append((i, i + 1))
edges.append((num_vertices - 1, 0))  # Connect the last vertex to the first one to close the loop

print(edges)


#%%


model = ifcopenshell.file()

# All projects must have one IFC Project element
project = run("root.create_entity", model, ifc_class="IfcProject", name="FactorySim Project")

# Geometry is optional in IFC, but because we want to use geometry in this example, let's define units
# Assigning without arguments defaults to metric units
length = run("unit.add_si_unit", model, unit_type="LENGTHUNIT", prefix="MILLI")
run("unit.assign_unit", model, units=[length])

# Let's create a modeling geometry context, so we can store 3D geometry (note: IFC supports 2D too!)
model3d = run("context.add_context", model, context_type="Model")
plan  = run("context.add_context", model, context_type="Plan")

# In particular, in this example we want to store the 3D "body" geometry of objects, i.e. the body shape
body = run("context.add_context", model, context_type="Model",
    context_identifier="Body", target_view="MODEL_VIEW", parent=model3d)

drawing = run("context.add_context", model,
    context_type="Plan", context_identifier="Annotation", target_view="PLAN_VIEW", parent=plan)

# Create a site, building, and storey. Many hierarchies are possible.
site = run("root.create_entity", model, ifc_class="IfcSite", name="IfcSite")
building = run("root.create_entity", model, ifc_class="IfcBuilding", name="IfcBuilding")
storey = run("root.create_entity", model, ifc_class="IfcBuildingStorey", name="IfcBuildingStorey")

# Since the site is our top level location, assign it to the project
# Then place our building on the site, and our storey in the building
run("aggregate.assign_object", model, relating_object=project, product=site)
run("aggregate.assign_object", model, relating_object=site, product=building)
run("aggregate.assign_object", model, relating_object=building, product=storey)

outer_polygon_points = [
    [0.0, 0.0, 0.0],
    [1000.0, 0.0, 0.0],
    [1000.0, 1000.0, 0.0],
    [0.0, 1000.0, 0.0],  # Closing point to complete the polygon
]

# Inner hole vertices for hole 1
inner_hole_points_1 = [
    [250.0, 250.0, 0.0],
    [750.0, 250.0, 0.0],
    [750.0, 750.0, 0.0],
    [250.0, 750.0, 0.0],  # Closing point to complete the hole
]

# Inner hole vertices for hole 2
inner_hole_points_2 = [
    [400.0, 400.0, 0.0],
    [600.0, 400.0, 0.0],
    [600.0, 600.0, 0.0],
    [400.0, 600.0, 0.0],
  # Closing point to complete the hole
]

# Create IfcCartesianPoints for outer polygon
outer_polygon_cartesian_points = [
    model.createIfcCartesianPoint(Coordinates=point) for point in outer_polygon_points
]

# Create IfcPolyLoop for outer polygon
outer_polygon_loop = ifcopenshell.create_entity(
    "IfcPolyLoop",
    Polygon=outer_polygon_cartesian_points
)
outer_polygon_loop = model.createIfcPolyLoop(Polygon=outer_polygon_cartesian_points)
# Create IfcFaceBound for outer polygon
outer_polygon_face_bound = model.createIfcFaceBound(Bound=outer_polygon_loop, Orientation=True)

# Create IfcFaceOuterBound for outer polygon
outer_polygon_face_outer_bound = model.createIfcFaceOuterBound(Bound=outer_polygon_loop,  Orientation=True)

# Create IfcFace for outer polygon
outer_polygon_face = model.createIfcFace(Bounds=[outer_polygon_face_outer_bound])


closed_shell = model.createIfcClosedShell(CfsFaces=[outer_polygon_face])
# Create IfcFacetedBrep
faceted_brep = model.createIfcFacetedBrep(Outer=closed_shell)

# Add the IfcFacetedBrep to the file
ifc_element = run("root.create_entity", model, ifc_class="IFCBUILDINGELEMENTPROXY", name="Test")
run("spatial.assign_container", model, relating_structure=storey, products=[ifc_element])
representation = model.createIfcShapeRepresentation(ContextOfItems=body, RepresentationIdentifier="Body", RepresentationType="Brep", Items=[faceted_brep])
#representation = run("geometry.add_representation", ifc_file, context=ifc_context, shape=shape, name=element.name)
run("geometry.assign_representation", model, product=ifc_element, representation=representation)

# Save the IFC file
model.write("working.ifc")
# %%


model = ifcopenshell.file()

# All projects must have one IFC Project element
project = run("root.create_entity", model, ifc_class="IfcProject", name="FactorySim Project")

# Geometry is optional in IFC, but because we want to use geometry in this example, let's define units
# Assigning without arguments defaults to metric units
length = run("unit.add_si_unit", model, unit_type="LENGTHUNIT", prefix="MILLI")
run("unit.assign_unit", model, units=[length])

# Let's create a modeling geometry context, so we can store 3D geometry (note: IFC supports 2D too!)
model3d = run("context.add_context", model, context_type="Model")
plan  = run("context.add_context", model, context_type="Plan")

# In particular, in this example we want to store the 3D "body" geometry of objects, i.e. the body shape
body = run("context.add_context", model, context_type="Model",
    context_identifier="Body", target_view="MODEL_VIEW", parent=model3d)

drawing = run("context.add_context", model,
    context_type="Plan", context_identifier="Annotation", target_view="PLAN_VIEW", parent=plan)

# Create a site, building, and storey. Many hierarchies are possible.
site = run("root.create_entity", model, ifc_class="IfcSite", name="IfcSite")
building = run("root.create_entity", model, ifc_class="IfcBuilding", name="IfcBuilding")
storey = run("root.create_entity", model, ifc_class="IfcBuildingStorey", name="IfcBuildingStorey")

# Since the site is our top level location, assign it to the project
# Then place our building on the site, and our storey in the building
run("aggregate.assign_object", model, relating_object=project, product=site)
run("aggregate.assign_object", model, relating_object=site, product=building)
run("aggregate.assign_object", model, relating_object=building, product=storey)

# Outer polygon vertices
outer_polygon_points = [
    [0.0, 0.0],
    [1000.0, 0.0],
    [1000.0, 1000.0],
    [0.0, 1000.0],
    [0.0, 0.0],  # Closing point to complete the polygon
]

# Inner hole vertices for hole 1
inner_hole_points_1 = [
    [130.0, 130.0],
    [250.0, 130.0],
    [250.0, 250.0],
    [130.0, 250.0],
    [130.0, 130.0],  # Closing point to complete the hole
]

# Inner hole vertices for hole 2
inner_hole_points_2 = [
    [400.0, 400.0],
    [600.0, 400.0],
    [600.0, 600.0],
    [400.0, 600.0],
    [400.0, 400.0],  # Closing point to complete the hole
]


# Create IfcCartesianPoints for outer polygon
outer_polygon_cartesian_points = [
    model.createIfcCartesianPoint(Coordinates=point) for point in outer_polygon_points
]

outer_polygon_loop = model.createIfcPolyLoop(Polygon=outer_polygon_cartesian_points)

# Create IfcFaceBound for outer polygon
outer_polygon_face_bound = model.createIfcFaceOuterBound(Bound=outer_polygon_loop, Orientation=False)

faces = [outer_polygon_face_bound]

# Create IfcFaceBound and IfcFaceOuterBound for each hole and add them to the outer face
for inner_hole_points in [inner_hole_points_1, inner_hole_points_2]:
    # Create IfcCartesianPoints for inner hole
    inner_hole_cartesian_points = [
        model.createIfcCartesianPoint(Coordinates=point) for point in inner_hole_points 
    ]

    # Create IfcPolyLoop for inner hole
    inner_hole_loop = ifcopenshell.create_entity(
        "IfcPolyLoop",
        Polygon=inner_hole_cartesian_points
    )

    # Create IfcFaceBound for inner hole
    inner_hole_face_bound = ifcopenshell.create_entity(
        "IfcFaceBound",
        Bound=inner_hole_loop,
        Orientation=False
    )

    # Add the inner hole to the outer face
    faces.append(inner_hole_face_bound)

# Create IfcFace for outer polygon
outer_polygon_face = model.createIfcFace(Bounds=faces)

# Create IfcClosedShell
closed_shell = model.createIfcClosedShell(CfsFaces=[outer_polygon_face])

# Create IfcFacetedBrep
faceted_brep = model.createIfcFacetedBrep(Outer=closed_shell)

# Add the IfcFacetedBrep to the file
ifc_element = run("root.create_entity", model, ifc_class="IFCBUILDINGELEMENTPROXY", name="Test")
run("spatial.assign_container", model, relating_structure=storey, products=[ifc_element])
representation = model.createIfcShapeRepresentation(ContextOfItems=body, RepresentationIdentifier="Body", RepresentationType="Brep", Items=[faceted_brep])
#representation = run("geometry.add_representation", ifc_file, context=ifc_context, shape=shape, name=element.name)
run("geometry.assign_representation", model, product=ifc_element, representation=representation)

# Save the IFC file
model.write("flat_holes.ifc")
# %%
