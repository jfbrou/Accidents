

import os

# Find the current working directory
path = os.getcwd()

# Create a folder that contains all figures
if os.path.isdir(os.path.join(path, 'Figures')) == False:
    os.mkdir('Figures')
figures_dir_path = os.path.join(path, 'Figures')


def draw_as_pdf(geometries, out_path):
    """
        Function to draw geometries on a pdf
    """

    if os.path.exists(out_path) == False:

        # plot
        fig, ax = plt.subplots()
        geometries.plot(ax=ax, color='teal', markersize=0.1)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_path, format='pdf')
        plt.close()

        # log
        logging.info(f'File created {out_path}')

    else:
        logging.info(f'Already exists {out_path}')



################################################################################
#                                                                              #
#                                 Produce PDFs                                 #
#                                                                              #
################################################################################

# write
out_path = os.path.join(figures_dir_path, 'accidents_2019.pdf')
draw_as_pdf(accidents, out_path)

# write
out_path = os.path.join(figures_dir_path, 'segments.pdf')
draw_as_pdf(road_segments, out_path)
