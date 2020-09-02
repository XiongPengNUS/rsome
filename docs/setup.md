# Setup
-------------------------

## Personal page (`<username>.github.io`)

### 1. Fork the repository
Fork the repository from [here](https://github.com/kbsezginel/gh-pages-template) by clicking the fork button on the top right corner.

### 2. Rename the repository
Go to `Settings -> Repository name` and write down `<username>.github.io`.

## Project page (`<username>.github.io/<myproject>`)

### 1. Fork the repository
Fork the repository from [here](https://github.com/kbsezginel/gh-pages-template) by clicking the fork button on the top right corner.

### 2. Enable GitHub Pages
Go to `Settings -> GitHub Pages` in your fork of the repository.
Under `source` select `master branch`.

# Customization
-------------------------

## Configuring the site
Edit the `_config.yml` file and change the site title and description.
Remove the `setup.md` file (optional).

## Adding pages
To add a new page you need to create a new markdown file with the name of your choice.
For example here `setup.md` file is used to generate content for the `setup` page.

### Layouts
There are currently two layouts: `default` and `full`.
In the `default` layout a sidebar containing navigation is present whereas in `full` layout the full page is reserved for page content.
The `full` layout can be selected by adding a yaml front matter to the markdown/html file:
```
---
layout: full
---
```

## Adding links to the navigation bar
The navigation bar can be controlled by modifying the `_config.yml` file.
Under navigation enter the title you would like to see on the sidebar and enter the relative link to that page.

## More customization
Take a look at minimal theme page to see all the theme options: [click here](https://pages-themes.github.io/minimal/).

**Serving from the `docs` folder**
> You can also serve the website from the `docs` folder. This is especially useful for project pages to keep the website files and project files separate. You basically need to keep all the files for the webpage in a folder named `docs` and in step 2 under `source` select `master branch/docs folder`. More info [here](https://help.github.com/articles/configuring-a-publishing-source-for-github-pages/).

# Resources and Tutorials
-------------------------

- [Minimal theme GitHub repository](https://github.com/pages-themes/minimal)

More tutorials on github pages, markdown and Jekyll can be found here:
- [GitHub pages basics](https://help.github.com/categories/github-pages-basics/)
- [GitHub basic writing and formatting syntax](https://help.github.com/articles/basic-writing-and-formatting-syntax/)
- [Markdown cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)
- [Setting up your GitHub Pages site locally with Jekyll](https://help.github.com/articles/setting-up-your-github-pages-site-locally-with-jekyll/)
- [Jekyll front matter](https://jekyllrb.com/docs/frontmatter/)
