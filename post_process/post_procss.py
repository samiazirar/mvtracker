# TODO use the Robot urdf model to mask the robot AND to get better visual image
# TODO Use heuristics to have it tracked also when occluded (end when removing)
# TODO Get the views and maskt eh hands robot etc
# TODO Track the hans and get object masks and 3D models
# TODO find static regions (whcih are e.g. occluded by hand) and make tem unchangeble and thenn rerun pipieline
# TODO reproject to depth afterwards also after pc creation to get better 
# TODO use vggt to get better features depth and "new" old views with better floe
# TODO allign the flow ..
# TODO Reprojectio error and average alignment error to filter out bad views
# TODO: Add 3D models of the objects for them to rerain rigid

# TODO: - notieren: -> photometric error von dem hintergrund
# ->  sowas wie gaussian splatting
# -> static reprojecting nur auf dasw as passiert
# -> point cloud prediction -> overparametrization auch hilfreich -> enforced consistency

# TODO Point cloud in map anything

# TODO Use Mapyanyting for upscalin and/or do self upscaling (then refinement with depth anything)
# TODO: Use filteres upscaled AND or low Res image and Reproject to be used in MapAnything for depth completion(or just reproject )