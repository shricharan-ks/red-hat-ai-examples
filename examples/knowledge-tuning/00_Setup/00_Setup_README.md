# Setup for the Knowledge Tuning example

To get started, follow these steps to set up the working environment on your Red Hat OpenShift AI cluster and complete the configuration for the Knowledge Tuning example:
[Heading IDs](#heading-ids)

1. [Create a project](#create-a-project)
2. [Create a workbench](#create-a-workbench)
3. [Clone the example Git repository](#clone-the-example-git-repository)
4. [Set up your JupyterLab environment](#set-up-your-jupyterlab-environment)

## Create a project

To implement a data science workflow, you must create a project. Projects help your team to organize and work together on resources within separated namespaces. From a project you can create many workbenches, each with their own IDE environment (for example, JupyterLab), and each with their own connections and cluster storage. In addition, the workbenches can share models and data with pipelines and model servers.

### Prerequisites

* You have logged in to Red Hat OpenShift AI.

### Procedure

. On the navigation menu, select *Projects*. This page lists any existing projects that you have access to.

. Click *Create project*. 

. In the *Create project* modal, enter a display name and description.

. Click *Create*.

### Verification

You can see your project's initial state. To view information about the project components and access permissions, click a tab:

- **Workbenches** are instances of your development and experimentation environment. They typically contain individual development environments (IDEs), such as JupyterLab, RStudio, and Visual Studio Code.

- **Pipelines** contain the data science pipelines which run within the project.

- **Deployments** for quickly serving a trained model for real-time inference. You can have many model servers per data science project. One model server can host many models.

- **Cluster storage** is a persistent volume that retains the files and data you're working on within a workbench. A workbench has access to one or more cluster storage instances.

- **Connections** contain required configuration parameters for connecting to a data source, such as an S3 object bucket.

- **Permissions** define which users and groups can access the project.


## Create a workbench

A workbench is an instance of your development and experimentation environment. When you create a workbench, you select a workbench image that has the tools and libraries that you need for developing models. 

### Prerequisites

- You created a project.

### Procedure

1. Navigate to the project detail page for the project that you created in **Create a project**.

2. Click the **Workbenches** tab, and then click **Create workbench**.

3. Fill out the name and description.

Red Hat OpenShift AI provides several supported workbench images. In the **Workbench image** section, you can select one of the default images or a custom image that an administrator has set up for you. The **Tensorflow** image has the libraries needed for this example.

4. Select the latest **Tensorflow** image.
<!-- edit for this example
-->

5. Choose a small deployment.

6. If your OpenShift cluster has available GPUs, the **Create workbench** form includes an **Accelerator** option. Select **None**. This example does not require any GPUs.
<!-- edit for this example
-->
7. Leave the default environment variables and storage options.

8. For **Connections**, click **Attach existing connection**.

9. Select `My Storage` (the object storage that you configured earlier) and then click **Attach**.

10. Click **Create workbench**.

### Verification

In the **Workbenches** tab for the project, the status of the workbench changes from `Starting` to `Running`.

NOTE: If you made a mistake, you can edit the workbench to make changes.

## Clone the example Git repository

The JupyterLab environment is a web-based environment, but everything you do inside it happens on Red Hat OpenShift AI and is powered by the OpenShift cluster. This means that, without having to install and maintain anything on your own computer, and without using valuable local resources such as CPU, GPU and RAM, you can conduct your work in this powerful and stable managed environment.

### Prerequisites

You created a workbench, as described in *Creating a workbench*.

### Procedure

1. Click the link for your workbench. If prompted, log in and allow JupyterLab to authorize your user.

Your JupyterLab environment window opens.

The file-browser window shows the files and folders that are saved inside your own personal space in OpenShift AI.

2. Bring the content of this example inside your JupyterLab environment:

a. On the toolbar, click the *Git Clone* icon:

b. Enter the following example Git *https* URL:

   https://github.com/red-hat-ai-examples/knowledge-tuning

c. Select the *Include submodules* option, and then click *Clone*.

d. In the file browser, double-click the newly-created *knowledge-tuning* folder.

e. In the left navigation bar, click the *Git* icon, and then click *Current Branch* to expand the branches and tags selector panel.

f.  On the *Branches* tab, in the *Filter* field, enter *{git-version}*.
<!-- should they use main for the git-version?
-->

g. Select *origin/{git-version}*. 

The current branch changes to *{git-version}*.


### Verification

In the file browser, view the notebooks that you cloned from Git.

## Set up your JupyterLab environment 

In JupyterLab, open the [00_Setup.ipynb](./00_Setup.ipynb) notebook and follow the instructions within the notebook.

Congratulations! Your workbench is configured and ready for the knowledge training example. The notebooks and supporting README files provide details about each step in the knowledge training workflow.

### Next step

[Data Processing](./01_Data_Preprocessing/README.md)


[def]: #Se