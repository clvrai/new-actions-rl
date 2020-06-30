import os

import envs.block_stack.utils as utils


def get_xml_id(polygon, names):
    i = 0
    while '{}_{}'.format(polygon, i) in names:
        i += 1
    name = '{}_{}'.format(polygon, i)
    return name

class XML:
    def __init__(self, asset_path='assets/stl/', shape_names=[], place2d=False,
            cam_pos=None, no_ground=False, contacts_off=False):
        self.asset_path = asset_path
        self.names = []
        self.shape_names = shape_names
        self.assets_mesh = []
        self.assets_material = [{'name': 'wall_visible', 'rgba': '.9 .9 .9 1'}, {
            'name': 'wall_invisible', 'rgba': '.9 .9 .9 0'}]
        self.meshes = []
        # <size njmax="500" nconmax="100"/>
        # Existing
            # <option timestep='0.01' integrator="RK4"  />
        self.contacts_off = contacts_off
        self.base = '''
<mujoco>
    <asset>
       {}
       {}
    </asset>
    <compiler angle='radian'/>

    <option timestep='0.01' />
    <size njmax="1500" nconmax="200"/>

    <visual>
      <map znear="0.0001"/>
    </visual>

   <worldbody>
      %s
      <light diffuse='1.5 1.5 1.5' pos='0 4 4' dir='0 -1 0'/>
      <light diffuse='1.5 1.5 1.5' pos='0 4 1' dir='0 -1 0'/>

      <geom name='wall_left'  type='box' pos='-3 -1 0' euler='0 0 0' size='0.1 2 4' material='wall_invisible'/>
      <geom name='wall_right'  type='box' pos='-3 -1 0' euler='0 0 0' size='0.1 2 4' material='wall_invisible'/>
      <geom name='wall_back'  type='box' pos='0 -4 2' euler='0 0 0' size='4 0.1 4' material='wall_invisible'/>
      <geom name='wall_floor'  type='plane' pos='0 0 -.5' euler='0 0 0' size='4 2 0.1' material='%s'/>

      {}

   </worldbody>
</mujoco>
'''
        if place2d:
            cam_y = 7
            cam_z = 2
            angle = -1.57
        else:
            cam_y = 4
            cam_z = 5
            angle = -0.8

            # To get a rendering of each object
            #cam_z = 4
            #angle = -1.1

        # The more positive the angle the more it is angled down

        if no_ground:
            ground_str = 'wall_invisible'
        else:
            ground_str = 'wall_visible'

        if cam_pos is None:
            cam_str = "<camera name='fixed' pos='0 %.2f %.2f' euler='%.2f 0 0'/>"
            cam_str = cam_str % (cam_y, cam_z, angle)
            self.cam_lookat = False
        else:
            cam_str = "<camera name='custom' mode='targetbody' pos='%.4f %.4f %.4f' target='{}'/>"
            cam_str = cam_str % (cam_pos[0], cam_pos[1], cam_pos[2])
            self.cam_lookat = True

        self.base = self.base % (cam_str, ground_str)



    def get_unique_name(self, polygon):
        name = get_xml_id(polygon, self.names)
        self.names.append(name)
        return name

    def add_asset(self, name, polygon, scale):
        self.assets.append({'name': name, 'polygon': polygon, 'scale': scale})

    def add_mesh(self, polygon, scale=1, pos=[0, 0, 0], axangle=[1, 0, 0], rgba=[1, 1, 1, 1], force=[0, 0, 0], name=None):
        if name is None:
            name = self.get_unique_name(polygon)
        else:
            self.names.append(name)

        scale_rep = self.__rep_vec([scale, scale, scale])
        pos_rep = self.__rep_vec(pos)
        quat = utils.axangle_to_quat(axangle)
        quat_rep = self.__rep_vec(quat)

        rgba_rep = self.__rep_vec(rgba)
        if name.split('_')[0] not in self.shape_names:
            self.assets_mesh.append(
                {'name': name, 'polygon': polygon, 'scale': scale_rep})
        self.assets_material.append({'name': name, 'rgba': rgba_rep})

        self.meshes.append({
            'name': name,
            'polygon': polygon,
            'pos': pos_rep,
            'quat': quat_rep,
            'force': force,
            'xscale': scale,
            'scale': scale_rep,
            'xrgba': rgba,
            'material': name})

        return name


    def __rep_vec(self, vec):
        vec = [str(v) for v in vec]
        return ' '.join(vec)


    def get_body_str(self):

        # add_xml = 'solref="0.02 1.3"'
        #add_xml = 'friction="2 0.5 0.0001"'
        add_xml = 'condim="6" friction="2 2 0.25"'

        if self.contacts_off:
            body_base = '''
            <body name='{}' pos='{}' quat='1 0 0 0'>
              <joint type='free' name='{}'/>
              <geom name='{}' type='mesh' mesh='{}' pos='0 0 0' contype='0' conaffinity='0' quat='{}' material='{}' {}/>
            </body>
          '''
            shape_base = '''
            <body name='{}' pos='{}' quat='1 0 0 0'>
              <joint type='free' name='{}'/>
              <geom name='{}' type='{}' size='{}' pos='0 0 0' contype='0' conaffinity='0' quat='{}' material='{}' {}/>
            </body>
          '''
        else:
            body_base = '''
            <body name='{}' pos='{}' quat='1 0 0 0'>
              <joint type='free' name='{}'/>
              <geom name='{}' type='mesh' mesh='{}' pos='0 0 0' quat='{}' material='{}' {}/>
            </body>
          '''
            shape_base = '''
            <body name='{}' pos='{}' quat='1 0 0 0'>
              <joint type='free' name='{}'/>
              <geom name='{}' type='{}' size='{}' pos='0 0 0' quat='{}' material='{}' {}/>
            </body>
          '''



        body_list = []
        for m in self.meshes:
            shape_type = m['name'].split('_')[0]
            if shape_type in self.shape_names:
                use_str = shape_base.format(m['name'], m['pos'],
                        m['name'], m['name'], shape_type, m['scale'],
                        m['quat'], m['material'], add_xml)
            else:
                use_str = body_base.format(m['name'], m['pos'],
                        m['name'], m['name'], m['name'], m['quat'], m['material'], add_xml)
            body_list.append(use_str)
        body_str = '\n'.join(body_list)
        return body_str

    def get_asset_mesh_str(self):
        asset_base = '<mesh name="{}" scale="{}" file="{}"/>'

        asset_list = [asset_base.format(
            a['name'], a['scale'],
            os.path.join(self.asset_path, a['polygon'] + '.stl'))
            for a in self.assets_mesh]

        asset_str = '\n'.join(asset_list)
        return asset_str

    def get_asset_material_str(self):
        asset_base = '<material name="{}" rgba="{}" specular="0" shininess="0" emission="0.25"/>'

        asset_list = [asset_base.format(
            a['name'], a['rgba'])
            for a in self.assets_material]

        asset_str = '\n'.join(asset_list)
        return asset_str

    def instantiate(self):
        if self.cam_lookat:
            xml_str = self.base.format(self.get_asset_mesh_str(),
                    self.get_asset_material_str(), self.names[0], self.get_body_str())
        else:
            xml_str = self.base.format(self.get_asset_mesh_str(),
                    self.get_asset_material_str(), self.get_body_str())
        return xml_str

    def apply_forces(self, sim):
        for mesh in self.meshes:
            mesh_name = mesh['name']
            force = mesh['force']
            mesh_ind = sim.model._body_name2id[mesh_name]
            sim.data.xfrc_applied[mesh_ind, :3] = force
        sim.step()
        sim.data.xfrc_applied.fill(0)
