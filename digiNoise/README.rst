Downloads
---------
Running subroutines found in this folder, in turn, require functions found in the irt package.
Following are the steps to add irt to your matlab path:

.. code-block:: bash

    $ cd ct-recon/digiNoise
    $ wget https://github.com/smuzd/LD-CT-simulation/blob/master/I0.mat?raw=true
    $ mv I0.mat\?raw\=true I0.mat
    $ mv I0* ./data/matfiles/ 
    $ cd ../..
    $ wget https://web.eecs.umich.edu/~fessler/irt/fessler.tgz
    $ tar -xvzf fessler.tgz
    $ mv ct-recon/error_analysis/cho_lcd/fbp2_window.m irt/fbp/
    $ mv irt ct-recon/ # move the irt package inside your ct-recon copy
    $ cd ct-recon/irt
    $ matlab 
    >> setup
    >> cd ../digiNoise

DEMO on NPS validation of our noise model against LDGC using water cylinder
-----------------------------------------------------------------------------

.. code-block:: matlab

    demo_nps_val_on_ldgc

DEMO on CT realizations using virtual CCT189 replica required for an `observer model based performance testing <https://github.com/prabhatkc/ct-recon/tree/main/error_analysis/cho_lcd#lcd-on-ldct-acquisition>`_
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

.. code-block:: matlab

    demo_nps_val_on_ldgc

DEMO on NPS validation of our noise model against LDGC using CT realizations from an XCAT
-----------------------------------------------------------------------------------------

.. code-block:: matlab

    demo_modified_xcat_realizations

DEMO on performing simulated CT reconstruction using acr's virtual module 1 replica
-----------------------------------------------------------------------------------------

.. code-block:: matlab

    demo_fbp_on_acr_module1