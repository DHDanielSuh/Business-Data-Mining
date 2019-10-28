def save_recog_results_to_xml(self, veh_feats):
    if self.check_tag_dir is None:
        return

    for veh_feat in veh_feats:
        veh_uri = veh_feat['veh_uri']
        plt_uri = veh_feat['plt_uri']
        plt_num = veh_feat['plt_num']

        img = cv2.imread(veh_uri, cv2.IMREAD_UNCHANGED)
        if img is None:
            self.logger.error(' no file {}'.format(veh_uri))
            return

        file_name = os.path.split(veh_uri)[-1]
        try:
            shutil.copyfile(veh_uri, os.path.join(self.check_tag_dir, file_name))
        except Exception as e:
            self.logger.error(
                ' copy file from \'{}\' to \'{}\' fail : {}'.format(veh_uri, self.check_tag_dir, e))
            return

        height = img.shape[0]
        width = img.shape[1]
        depth = img.shape[2]

        file_loader = FileSystemLoader('template')
        environment = Environment(loader=file_loader, autoescape=True)
        annotation_template = environment.get_template('annotation.xml')

        abspath = os.path.abspath(veh_uri)

        template_parameters = {
            'path': abspath,
            'filename': os.path.basename(abspath),
            'folder': os.path.basename(os.path.dirname(abspath)),
            'width': width,
            'height': height,
            'depth': depth,
            'objects': []
        }

        template_parameters['objects'].append({
            'name': 'plate',
            'result': plt_num,
            'groundtruth': None,
        })

        file_name = file_name.rsplit('.', 1)[0] + '.xml'
        xml_path = os.path.join(self.check_tag_dir, file_name)
        with open(xml_path, 'w') as file:
            content = annotation_template.render(**template_parameters)
            file.write(content)
