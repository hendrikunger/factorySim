from shapely.affinity import  scale
from ifcopenshell.api import run
import ifcopenshell
import copy

def prepare_for_export(element_dict, bb):
    new_dict = copy.deepcopy(element_dict)
    for element in new_dict.values():
        element.poly = scale(element.poly, yfact=-1, origin=bb.centroid)
        polybbox = element.poly.bounds
        element.origin = (polybbox[0], polybbox[1])

    return new_dict

def write_ifc_class(model, ifc_context, ifc_class, element_dict):
    elements = []

    for element in element_dict.values():
        ifc_element = run("root.create_entity", model, ifc_class=ifc_class, name=element.name)
        #matrix[:,3][0:3] = (element.origin[0]/1000, element.origin[1]/1000, 0)
        run("geometry.edit_object_placement", model, product=ifc_element)
        #run("geometry.edit_object_placement", model, product=ifc_element, matrix=matrix, is_si=True)
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
        run("geometry.assign_representation", model, product=ifc_element, representation=representation)
        elements.append(ifc_element)
    return elements