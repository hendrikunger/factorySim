# %%
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon, GeometryCollection, box
from shapely.affinity import translate, rotate, scale
from shapely.ops import unary_union
import random
import math
import ifcopenshell
from factorySim.Helpers.MFO import MFO



# %%

elementName = "IFCBUILDINGELEMENTPROXY"
#elementName = "IFCWALL"
randomMF=None

ifc_file = ifcopenshell.open("Basic.ifc")

elementlist = []
elements = []
if(randomMF):
    elements = random.choices(ifc_file.by_type(elementName), k=random.randint(2, randomMF))
else:
    elements = ifc_file.by_type(elementName)
for index, element in enumerate(elements):
    #get origin
    origin = element.ObjectPlacement.RelativePlacement.Location.Coordinates
    #element.ObjectPlacement.RelativePlacement.Axis.DirectionRatios[0]
    #element.ObjectPlacement.RelativePlacement.RefDirection.DirectionRatios[0]

    #get rotation
    x = element.ObjectPlacement.RelativePlacement.RefDirection.DirectionRatios[0]
    y = element.ObjectPlacement.RelativePlacement.RefDirection.DirectionRatios[1]
    rotation = math.atan2(y,x)

    my_uuid = ""
    if(randomMF):
        my_uuid= "_" + str(index)

    #create MFO Object
    # mfo_object = MFO(gid=element.GlobalId, 
    #     name=element.Name + my_uuid,
    #     origin_x=origin[0],
    #     origin_y=origin[1],
    #     origin_z=origin[2],
    #     rotation=rotation)

    #points = element.Representation.Representations[0].Items[0].Outer.CfsFaces[0].Bounds[0].Bound.Polygon
    #Always choose Representation 0
    items = element.Representation.Representations[0].Items

    #Parse imported data into MFO Object (Machine)

  
    result = []

    for item in items:
        faces = item.Outer.CfsFaces
        facelist = []

        for face in faces:
            exterior = []
            interior = []
            bounds = face.Bounds
            for bound in bounds:
                type = bound.get_info()['type']
                points = bound.Bound.Polygon
                #Remove z Dimension of Polygon Coordinates
                pointlist = [(point.Coordinates[0], point.Coordinates[1]) for point in points ]
                #Check if face is a surface or a hole
                if (type == "IfcFaceBound"):
                    interior.append(pointlist)
                    #print("         Hole")
                else:
                    #Case surface
                    exterior = pointlist
                    #print("         Face")

            facelist.append(Polygon(exterior, interior))
            #print(f"      Faces: {facelist}")



        result.append(MultiPolygon(facelist))
        #print(f"   Result: {result}")
        #print("   ------------------------------------")
    singleElement = unary_union(result)
    if singleElement.type != "MultiPolygon":
        singleElement = MultiPolygon([singleElement])
    print(singleElement)
    singleElement = translate(singleElement, origin[0], origin[1])
    singleElement = rotate(singleElement, rotation, origin=(origin[0], origin[1]), use_radians=True)
    
    #create MFO Object       
    mfo_object = MFO(gid=element.GlobalId, 
                name=element.Name + my_uuid,
                origin=(origin[0], origin[1]),
                poly=singleElement)

    elementlist.append(mfo_object)
    #print(f"Elementlist: {elementlist}")
    #print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n\n") 

    #final = MultiPolygon(result)
    #     mfo_object.close_Item()
    # mfo_object.updatePosition() 

    


# %%
elementlist[1].poly
# %%
allElements = unary_union([x.poly for x in elementlist])
allElements
# %%
boundingBox = allElements.bounds      
min_value_x = boundingBox[0]     
max_value_x = boundingBox[2]     
min_value_y = boundingBox[1]     
max_value_y = boundingBox[3] 

scale_x = 400 / (max_value_x - min_value_x)
scale_y = 400 / (max_value_y - min_value_y)
scale = min(scale_x, scale_y)

for m in elementlist:
    m.scale_Points(scale, scale, -min_value_x, -min_value_y)

#%%
for element in elementlist:
    for poly in element.poly.geoms:
        for loop in poly.interiors:
            for point in loop.coords:
                print(point[0], point[1])
        


# %%
elementlist[0].poly.representative_point().y
# %%


# %%
