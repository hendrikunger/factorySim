from shapely.affinity import  scale, rotate, translate
from ifcopenshell.api import run
import numpy as np
import ifcopenshell
import copy
import requests

def prepare_for_export(element_dict, bb):
    """This function moves the geometry to the center of the bounding box and flips the y-axis to match the IFC coordinate system

    Args:
        element_dict (_type_): dict of factory objects to be exported
        bb (_type_): bounding box of the factory

    Returns:
        _type_: dict of transformed factory objects
    """
    new_dict = copy.deepcopy(element_dict)
    for element in new_dict.values():
        element.poly = rotate(element.poly, element.rotation, origin="center", use_radians=True)
        #Fix mirroring 
        element.poly = scale(element.poly, yfact=-1, origin=bb.centroid)
        #Set origin to the center of the bounding box
        bounds = element.poly.bounds
        element.poly = translate(element.poly, 
                                 xoff=-(bounds[0] + (bounds[2] - bounds[0])/2), 
                                 yoff=-(bounds[1] + (bounds[3] - bounds[1])/2)
        )
      
    return new_dict

def write_ifc_class(model, ifc_context, ifc_class, element_dict, factoryheight):
    elements = []

    for element in element_dict.values():
        ifc_element = run("root.create_entity", model, ifc_class=ifc_class, name=element.name)
        matrix = np.eye(4)
        #transform element.origin from y- down coordinates to y+ up coordinates
        newOrigin = (element.origin[0], factoryheight - element.origin[1])
        # this rotates around the coordinate system origin, for this reason prepare_for_export moves the geometry to
        #the center of the bounding box
        matrix = ifcopenshell.util.placement.rotation(element.rotation, "Z", is_degrees=False) @ matrix
        #Set translation position to the new origin
        matrix[:,3][0:3] = ((newOrigin[0] + element.width/2)/1000, ((newOrigin[1] - element.height/2)/1000), 0)
        ifcopenshell.api.geometry.edit_object_placement(model, product=ifc_element, matrix=matrix, is_si=True)

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


def map_factorySpace_to_unit(x_mm, min_mm, max_mm):
    return (x_mm - min_mm) / (max_mm - min_mm)

def map_unit_to_factorySpace(x_unit, min_mm, max_mm):
    return x_unit * (max_mm - min_mm) + min_mm

def check_internet_conn():
# initializing URL
    url = "https://www.google.de"
    timeout = 3
    try:
        # requesting URL
        request = requests.get(url,
                            timeout=timeout)
        return True
    
    # catching exception
    except (requests.ConnectionError,
            requests.Timeout) as exception:
        return False