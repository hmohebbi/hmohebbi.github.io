---
layout: default
title: Blog
permalink: /blog/
---

{% for post in site.posts %}
	{% unless post.hidden %}
  * [ {{ post.title }} ]({{post.url}})
  	{% endunless %}
{% endfor %}