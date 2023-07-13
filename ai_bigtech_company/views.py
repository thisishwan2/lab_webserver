import io
import zipfile

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import requests
from bs4 import BeautifulSoup
import time
import pandas as pd

from matplotlib import pyplot as plt

from django.http import HttpResponse

from django.shortcuts import render

from ai_bigtech_company.forms import MyForm




# 비슷한 토픽끼리는 하나의 칼럼이 될 수 있도록 일일히 비슷한 토픽끼리 묶는 작업을 하였음
target = ['machinelearning', 'package', 'zsh', 'zerossl', 'youtube', 'xcode', 'x64', 'wysiwyg', 'workflow', 'windows',
          'weixin', 'wechat', 'web', 'vue',
          'vpn', 'vision', 'vim', 'vedio', 'version', 'util', 'ui', 'typescript', 'twitter', 'touch', 'test',
          'terminal', 'tensorflow',
          'telegram', 'task', 'system', 'swift', 'swagger', 'svg', 'style', 'static', 'state', 'ssh', 'sql', 'spring',
          'sort', 'software',
          'snippets', 'server', 'security', 'search', 'scikit-learn', 'rfc', 'rest', 'redis', 'react', 'queue',
          'pytorch', 'python', 'protoc', 'progress',
          'programming', 'posix', 'php', 'performance', 'openwrt', 'note', 'nodejs', 'neural', 'nlp', 'monitoring',
          'microservice', 'material', 'macos', 'linux',
          'leetcode', 'language', 'kotlin', 'json', 'javascript', 'ios', 'interview', 'http', 'html', 'headless',
          'graphql', 'git', 'frontend', 'font', 'flutter',
          'firefox', 'file', 'face-swap', 'docker', 'distributed', 'design', 'deep-learning', 'data', 'css', 'cloud',
          'chrome', 'automation', 'api', 'angular',
          'android', 'algorithm', 'pull-request', 'public', 'pixi', 'open-source', 'object-detection', 'multi-platform',
          'mini', 'low-code', 'localstorage', 'loading',
          'kubernetes', 'koa', 'image', 'humans', 'game', 'free', 'detection', 'drag', 'document', 'devops',
          'dashboard', 'crypto', 'cpp', 'continuous-delivery', 'command', 'cheatsheet',
          'certificate', 'business', 'build', 'bootstrap', 'awesome', 'audio', 'apple', 'apache', 'ansi', 'animation',
          'alibaba', 'admin', 'activejob'
          ]

string_list = [['machinelearning-python', 'ml', 'machine-translation', 'machine-learning-from-scratch',
                'machine-learning-algorithms', 'machine-learning'], ['packages', 'package-manager', 'packaging'],
               ['zsh-theme', 'zsh-prompt', 'zsh-configuration', 'zsh-autosuggestions'],
               ['zero-copy', 'zero-configuration'], ['youtube-live'], ['xcode10'], ['x86-64', 'x86'],
               ['wysiwyg-editor'],
               ['workflow-automation', 'works-with-clojurescript', 'works-with-codespaces', 'works-with-construct',
                'works-with-elm', 'works-with-flutter', 'works-with-gatsby', 'works-with-mint', 'works-with-phaser',
                'works-with-react', 'works-with-reason', 'works-with-svelte', 'works-with-vue', 'works-with-yew'],
               ['windowing', 'windows-10', 'windows-console', 'windows-terminal', 'windowscontainers'],
               ['weixin-plugin'], ['wechat-macos', 'wechat-mini-program', 'wechat-plugin', 'wechat-weapp'],
               ['webxr', 'webworkers', 'webview', 'webtorrent', 'webstorm', 'websql', 'websockets', 'websocket-server',
                'websocket-compression', 'websocket-client', 'websocket', 'website', 'webservices', 'webscraping',
                'webrtc', 'webpack5', 'webpack4', 'webpack2', 'webpack', 'webp', 'webm', 'webgpu', 'webgl2', 'webgl',
                'webframework', 'webfont', 'webfinger', 'webdriver', 'webdevelopment', 'webdav', 'webcomponents',
                'webbrowser', 'webaudio', 'webassembly', 'webapp', 'webapi', 'web-terminal', 'web-server',
                'web-performance', 'web-framework', 'web-development', 'web-components', 'web-clipper', 'web-audio',
                'web-application-framework', 'web-application', 'web-app', 'web', 'sdwebimage', 'react-native-web',
                'node-webkit', 'mobile-web', 'ipfs-web'],
               ['vuex', 'vuetifyjs', 'vuetify', 'vuepress', 'vuejs3', 'vuejs2', 'vuejs', 'vue3', 'vue2', 'vue-router',
                'vue-material', 'vue-components', 'vue-component', 'vue-cli', 'vue-admin', 'ant-design-vue'],
               ['vpn-server', 'vpn-client'],
               ['visual-studio', 'visualizer', 'visualizations', 'visualization', 'visual-studio-code-theme',
                'visual-studio-code-extension', 'visual-studio-code', 'visual-studio', 'data-visualization'],
               ['vimrc', 'vimeo', 'vim-plugins', 'vim-plugin', 'vim-distribution', 'vim-configuration', 'vim'],
               ['videojs', 'videocache', 'video-recording', 'video-player', 'video-conferencing', 'video-communication',
                'video', 'scalable-video-conferences', 'html5-video', 'explanatory-math-videos', 'deep-video',
                '3b1b-videos'],
               ['versioning', 'version-manager'], ['utils', 'utility-classes', 'utilities'],
               ['ui-library', 'ui-kit', 'ui-design', 'ui-components', 'redis-gui-client', 'gui-framework', 'gui-app'],
               ['typescript-playground', 'typescript-framework', 'typescript-definitions', 'typescript'],
               ['twitter-streaming', 'twitter-cards', 'twitter', 'tweet'],
               ['touchscreen', 'touch-events', 'touch'],
               ['unit-testing', 'tests', 'testing-tools', 'testing', 'test-suite', 'test-runners', 'test-runner',
                'test-framework', 'test-automation', 'performance-testing', 'pentesting-windows', 'pentesting',
                'pentesters', 'pentest', 'penetration-testing', 'painless-javascript-testing', 'load-testing',
                'javascript-tests', 'interview-test', 'end-to-end-testing', 'e2e-tests', 'e2e-testing', 'api-testing',
                'api-test'],
               ['windows-terminal', 'web-terminal', 'terminals', 'terminal-themes', 'terminal-schemes',
                'terminal-emulators', 'terminal-color', 'terminal', 'osx-terminal-themes', 'lxterminal'],
               ['tensorflow-lite', 'tensorflow', 'tensor'], ['telegram-solution', 'telegram-desktop'],
               ['task-scheduler', 'task-runner', 'task-manager'],
               ['systems-science', 'systems', 'system-design', 'system', 'recommendedsystem', 'operating-system',
                'management-system', 'fuse-filesystem', 'filesystem', 'embedded-systems', 'distributed-systems',
                'design-systems', 'design-system', 'database-systems', 'content-management-system', 'build-system'],
               ['swiftyjson', 'swiftui', 'swiftpm', 'swift4-2', 'swift-programming', 'swift-package-manager',
                'swift-library', 'swift-language', 'swift-framework', 'swift-extensions', 'swift', 'server-side-swift',
                'rxswift', 'reactiveswift', 'openstack-swift', 'json-parsing-swift', 'ios-swift', 'awesome-swift',
                'apple-swift'],
               ['swagger2', 'swagger-ui', 'swagger-js', 'swagger-api'],
               ['svgo', 'svg-sprites', 'svg-path', 'svg-icons'],
               ['stylesheets', 'stylesheet-language', 'styleguide', 'styled-components', 'style-linter', 'style-guide'],
               ['static-site-generator', 'static-code-analysis', 'static-analysis'],
               ['statechart', 'state-tree', 'state-management', 'state-machine'],
               ['ssh-tunnel', 'ssh-server', 'ssh-agent'],
               ['websql', 'sqlserver', 'sqlmap', 'sqlite', 'sql-server', 'sql-injection', 'sql-editor', 'sql',
                'spark-sql', 'postgresql', 'nosql', 'mysql-compatibility', 'mysql', 'mssql', 'microsoft-sql-server'],
               ['springs', 'springmvc', 'springcloud-config', 'springcloud', 'springbootadmin', 'springboot-shiro',
                'springboot', 'spring-session', 'spring-security', 'spring-physics', 'spring-framework',
                'spring-data-jpa', 'spring-cloud-core', 'spring-cloud-alibaba', 'spring-cloud', 'spring-cache',
                'spring-boot-upload-file', 'spring-boot-mongodb', 'spring-boot-mail', 'spring-boot-learning',
                'spring-boot-examples', 'spring-boot-demo', 'spring-boot-2', 'spring-boot', 'spring', 'react-spring'],
               ['sorts', 'sorting-algorithms', 'sortable'],
               ['software-engineering', 'software-engineer', 'software-development', 'software-analysis'],
               ['snippets-library', 'snippets-collection', 'snippets'],
               ['websocket-server', 'web-server', 'vpn-server', 'ssh-server', 'sqlserver', 'sql-server',
                'serverless-functions', 'serverless-framework', 'serverless-architectures', 'serverless',
                'server-side-swift', 'server-side-rendering', 'server-rendering', 'server', 'parse-server', 'observer',
                'microsoft-sql-server', 'jumpserver', 'http-server', 'graphql-server', 'dns-server', 'dhcp-server',
                'dev-server', 'api-server'],
               ['spring-security', 'security-researchers', 'security-checklist', 'security-book', 'security',
                'macos-security', 'macbook-security', 'docker-security', 'application-security'],
               ['site-search', 'silver-searcher', 'security-researchers', 'searches', 'search-in-text',
                'search-engines', 'search-engine', 'search', 'research', 'jobsearch', 'jobs-search',
                'hyperparameter-search', 'entreprise-search', 'elasticsearch', 'distributed-search-engine'],
               ['scikit-learn-python', 'scikit-learn'],
               ['rfc-8259', 'rfc-7159', 'rfc-7049', 'rfc-6902', 'rfc-6901', 'rfc-6455'],
               ['restful-api', 'restful', 'rest-api', 'rest', 'postgrest', 'api-rest', 'admin-on-rest'],
               ['redis-gui-client', 'redis-gui', 'redis-cluster', 'redis-client'],
               ['works-with-react', 'reactjs', 'reactivex', 'reactiveui', 'reactiveswift', 'reactivecocoa',
                'reactive-streams', 'reactive-programming', 'reactive', 'react16', 'react-tutorial', 'react-spring',
                'react-router', 'react-navigation', 'react-native-web', 'react-native', 'react-interview-questions',
                'react-hooks', 'react-components', 'react-component', 'react-boilerplate', 'react-apps', 'react-admin'],
               ['sqs-queue', 'queued-jobs', 'queue-workers', 'queue-tasks', 'queue', 'message-queue'],
               ['pytorch-tutorial', 'pytorch-transformers'],
               ['scikit-learn-python', 'python3', 'python-types', 'python-resources', 'python-requests',
                'python-library', 'python-interview-questions', 'python-framework', 'python-cheatsheet', 'python-3',
                'python', 'progress-bar-python'],
               ['protocol-compiler', 'protocol-buffers', 'protocol', 'protoc', 'ipfs-protocol',
                'homekit-accessory-protocol', 'acme-protocol'],
               ['progressmeter', 'progressbar', 'progress-bar-python', 'progress-bar'],
               ['swift-programming', 'reactive-programming', 'programming-tutorial', 'programming-ligatures',
                'programming-language', 'programming-interviews', 'programming-blogs', 'programming', 'programmer',
                'javascript-programming', 'functional-programming', 'dynamic-programming-algorithm',
                'dynamic-programming', 'concurrent-programming', 'competitive-programming'],
               ['posix-sh', 'posix-compliant-flags', 'posix-compliant'],
               ['phpunit', 'phpstorm', 'phpmailer', 'php8', 'php7', 'php-library', 'php-installation', 'php-framework',
                'php-applications', 'php', 'modern-php', 'designpatternsphp'],
               ['web-performance', 'performance-testing', 'performance-metrics', 'performance-analysis', 'performance',
                'high-performance', 'best-performance'],
               ['openwrt-zh-cn', 'openwrt-shadowsocksr-libev-full', 'openwrt-package', 'openwrt-feed'],
               ['notes', 'notebooks', 'notebook', 'note', 'jupyter-notebook', 'evernote'],
               ['nodemon', 'nodejs-framework', 'nodejs-development', 'nodejs', 'node-webkit', 'node-module', 'node-js',
                'node-framework', 'node'],
               ['neural-networks', 'neural-network', 'neural-nets', 'deep-neural-networks'],
               ['natural-language-understanding', 'natural-language-processing', 'natural-language-generation',
                'nlp-tasks', 'nlp-library'], ['monitoring', 'monitor', 'error-monitoring', 'docker-monitoring'],
               ['polyglot-microservices', 'microservices-architecture', 'microservices'],
               ['vue-material', 'training-materials', 'material-ui', 'material-theme', 'material-design-lite',
                'material-design-for-bootstrap', 'material-design', 'material-components', 'material',
                'bootstrap-material', 'angularjs-material'],
               ['wechat-macos', 'macosx', 'macos-setup', 'macos-security', 'macos-application'],
               ['ubuntu-linux', 'linux-resource', 'linux-kernel', 'linux-insides', 'linux-desktop', 'linux-command',
                'linux-app'],
               ['leetcode-solutions', 'leetcode-questions', 'leetcode-java', 'leetcode-golang', 'leetcode-go',
                'leetcode-c'], ['swift-language', 'stylesheet-language', 'ruby-language', 'programming-language',
                                'natural-language-understanding', 'natural-language-processing',
                                'natural-language-generation', 'multi-language', 'language-models', 'language-model',
                                'language-grammars', 'language-client', 'language', 'julia-language',
                                'esoteric-language'],
               ['kotlin-library', 'kotlin-android'],
               ['ubjson', 'swiftyjson', 'json-serializer', 'json-serialization', 'json-schema', 'json-pointer',
                'json-patch', 'json-parsing-swift', 'json-parsing-library', 'json-parser', 'json-merge-patch',
                'json-diff', 'json', 'fastjson'],
               ['painless-javascript-testing', 'javascript-tests', 'javascript-programming', 'javascript-motion-engine',
                'javascript-modules', 'javascript-library', 'javascript-interview-questions', 'javascript-framework',
                'javascript-engines', 'javascript-engine', 'javascript-compiler', 'javascript-closures',
                'javascript-best-practices', 'javascript-applications', 'javascript-algorithms', 'javascript',
                'es6-javascript', 'js-questions', 'js-ipfs', 'js'],
               ['ios-transition', 'ios-swift', 'ios-libraries', 'ios-lib', 'ios-framework', 'ios-animation'],
               ['technical-coding-interview', 'react-interview-questions', 'rails-interview',
                'python-interview-questions', 'programming-interviews', 'javascript-interview-questions', 'interviews',
                'interviewing', 'interview-test', 'interview-questions', 'interview-preparation', 'interview-prep',
                'interview-practice', 'interview', 'frontend-interview', 'front-end-interview', 'fe-interview',
                'coding-interviews', 'coding-interview', 'angularjs-interview-questions',
                'android-interview-questions'],
               ['httpurlresponse', 'https', 'httpie', 'httpclient', 'http2', 'http-server', 'http-proxy', 'http-client',
                'http', 'automatic-https'],
               ['htmlparser2', 'htmlparser', 'html5-video', 'html5-games', 'html5-game-development', 'html5-charts',
                'html5-canvas', 'html5-boilerplate', 'html5-audio', 'html5', 'html-template', 'html-questions'],
               ['headless-cms', 'headless-chrome', 'headless-browser'],
               ['graphql-server', 'graphql-relay', 'graphql-js', 'graphql-client', 'graphql-api', 'graphql',
                'apollographql'],
               ['hellogithub', 'gitops', 'gitlab', 'gitignore', 'github-pages', 'github-gist', 'github-api-v4',
                'github-api', 'github-actions', 'github', 'gitea', 'gitbook', 'git'],
               ['frontend-roadmap', 'frontend-interview', 'frontend-framework', 'frontend-development', 'frontend',
                'front-end-interview', 'front-end-development', 'front-end-developer-tool', 'front-end'],
               ['webfont', 'variable-fonts', 'ttf-fonts', 'patched-fonts', 'opentype-fonts', 'iconic-fonts',
                'icon-font', 'fonts', 'fontawesome', 'font-awesome'],
               ['works-with-flutter', 'flutter-plugins', 'flutter-plugin', 'flutter-examples', 'flutter-doc',
                'flutter-demo', 'flutter-apps'], ['firefox-extension', 'firefox-addon'],
               ['filesystem', 'files', 'file-uploads', 'file-uploader', 'file-upload', 'file-storage', 'file-sharing',
                'file-manager', 'file-management'],
               ['openfaceswap', 'faceswap', 'face-swap', 'face-recognition', 'face-images', 'face-detection',
                'deepfacelab', 'deepface', 'deep-face-swap'],
               ['dockerfile', 'docker-ui', 'docker-swarm', 'docker-security', 'docker-registry', 'docker-monitoring',
                'docker-machine', 'docker-image', 'docker-environment', 'docker-deployment', 'docker-container',
                'docker-composer', 'docker-compose', 'docker-api'],
               ['distributed-transactions', 'distributed-transaction', 'distributed-training', 'distributed-tracing',
                'distributed-systems', 'distributed-search-engine', 'distributed-messaging', 'distributed-locks',
                'distributed-database', 'distributed-configuration', 'distributed'],
               ['ui-design', 'system-design', 'material-design-lite', 'material-design-for-bootstrap',
                'material-design', 'designpatternsphp', 'design-tools', 'design-systems', 'design-system',
                'design-patterns-for-humans', 'design-patterns', 'design-pattern', 'design', 'api-design',
                'ant-design-vue', 'ant-design'],
               ['deepspeech', 'deeplearning', 'deepfakes', 'deepfacelab', 'deepface', 'deep-video',
                'deep-reinforcement-learning', 'deep-neural-networks', 'deep-networks', 'deep-learning-tutorial',
                'deep-learning', 'deep-io', 'deep-face-swap', 'creating-deepfakes'],
               ['datasets', 'datamatrix', 'datamapper', 'datagrip', 'databricks', 'databases', 'database-systems',
                'database-gui', 'database', 'data-viz', 'data-visualization', 'data-structures', 'data-structure',
                'data-scientists', 'data-science', 'data-mining', 'data-mapper', 'data-flow', 'data-fetching',
                'data-engineering', 'data-consistency', 'data-analytics', 'data-analysis'],
               ['functional-css', 'csse', 'css3-features', 'css3', 'css-tricks', 'css-reset', 'css-questions',
                'css-in-js', 'css-framework', 'css-effects', 'css-animations'],
               ['springcloud-config', 'springcloud', 'spring-cloud-core', 'spring-cloud-alibaba', 'spring-cloud',
                'private-cloud', 'nextcloud', 'neteasecloud', 'google-cloud-storage', 'google-cloud-functions',
                'cloudstorage', 'cloudnative', 'cloudmusic', 'cloudflare', 'cloud-storage', 'cloud-native',
                'cloud-management', 'cloud-computing'],
               ['headless-chrome', 'chrome-extensions', 'chrome-extension', 'chrome-devtools'],
               ['zsh-autosuggestions', 'workflow-automation', 'test-automation', 'home-automation', 'build-automation',
                'autosuggest', 'autopep8', 'autonomy', 'autonomous-vehicles', 'autonomous-driving', 'automl',
                'automation', 'automatic-https', 'automatic-api', 'automated', 'autolayout', 'autograd',
                'autocompletion', 'autocomplete', 'auto'],
               ['webscraping', 'webapi', 'swagger-api', 'strapi', 'scraping', 'restful-api', 'rest-api', 'public-apis',
                'openapi3', 'openapi-specification', 'openapi', 'homekit-api', 'hapi', 'graphql-api', 'github-api-v4',
                'github-api', 'fastapi', 'dom-apis', 'docker-api', 'automatic-api', 'apis', 'api-testing', 'api-test',
                'api-server', 'api-rest', 'api-manager', 'api-management', 'api-gateway', 'api-documentation',
                'api-design', 'api-client'],
               ['angularjs-material', 'angularjs-interview-questions', 'angularjs', 'angular9', 'angular2', 'angular10',
                'angular-dashboard-template', 'angular-components', 'angular-cli'],
               ['mpandroidchart', 'kotlin-android', 'androidx', 'android-ui', 'android-studio', 'android-library',
                'android-interview-questions', 'android-development', 'android-architecture'],
               ['sorting-algorithms', 'machine-learning-algorithms', 'javascript-algorithms', 'genetic-algorithm',
                'dynamic-programming-algorithm', 'algorithms-implemented', 'algorithms-datastructures',
                'algorithms-and-data-structures', 'algorithms', 'algorithm-competitions', 'algorithm-challenges'],
               ['pulltorefresh', 'pull-to-refresh', 'pull-request-review'],
               ['public-trackers', 'public-tracker', 'public-key-pinning'], ['pixiv', 'pixijs'],
               ['openwrt', 'openvpn', 'openssl', 'opensource', 'openpose', 'opengl', 'opendata', 'opencv',
                'openconnect', 'open-telemetry', 'open-source-project'],
               ['objectstorage', 'objective-c-library', 'objective-c', 'object-storage', 'object-oriented'],
               ['multithreading', 'multimedia', 'multiformats', 'multiboot', 'multi-platform', 'multi-person'],
               ['miniprogram', 'minimalist', 'minikube', 'minify', 'minifier', 'minification', 'miniapp'],
               ['low-code-plattform', 'low-code-development-platform', 'low-code'],
               ['localstack', 'localhost', 'localforage', 'local-development'],
               ['loading', 'loaders', 'load-generator', 'load-balancer'],
               ['kubernetes-ingress-controller', 'kubernetes-ingress'], ['koa2', 'koa-middleware'],
               ['imageloader', 'image-processor', 'image-processing', 'image-annotation'],
               ['human-pose-estimation', 'human-pose', 'human-behavior-understanding', 'forhumans'],
               ['puzzle-game', 'platform-game', 'not-the-game', 'games', 'gamedev', 'game-frameworks', 'game-engine',
                'game-development', 'game-capture', 'game', 'facebook-instant-games'],
               ['freeware-icon', 'freegate', 'freecodecamp', 'freebsd-vt', 'free-ssr', 'free-ss', 'free-software',
                'free-fanqiang'],
               ['object-detection', 'keypoint-detection', 'intrusion-detection', 'feature-detection'],
               ['dragging', 'draggable', 'drag-drop', 'drag-and-drop'], ['documentation-tool', 'documentation'],
               ['devops-roadmap'],
               ['hystrix-dashboard', 'dashboards', 'dashboard-templates', 'dashboard', 'admin-dashboard'],
               ['cryptography', 'cryptocurrency'], ['cpplint', 'cppcon', 'cpp11', 'cpp-library'],
               ['continuous-integration', 'continuous-deployment'],
               ['subcommands', 'commandline', 'command-line-tool', 'command-line', 'command-cobra'], ['cheatsheets'],
               ['certification', 'certificates', 'certificate-pinning'],
               ['businessintelligence', 'business-intelligence', 'business-analytics'],
               ['form-builder', 'buildpack', 'build-tools', 'build-tool', 'build-pipelines'],
               ['secure-boot', 'multiboot', 'jeecgboot', 'jeecg-boot', 'bootstrap5', 'bootstrap4-theme', 'bootstrap4',
                'bootstrap', 'bootable-usb'],
               ['awesomeness', 'awesome-public-datasets', 'awesome-lists', 'awesome-list'],
               ['audio-processing', 'audio-library'], ['apple-watch', 'apple-tv', 'apple'],
               ['apache-superset', 'apache-airflow'],
               ['strip-ansi', 'ansible', 'ansi-escape-codes', 'ansi-colors', 'ansi'],
               ['transition-animation', 'recyclerview-item-animation', 'animations', 'animation-library'],
               ['alibaba-oss', 'alibaba-middleware'],
               ['ngx-admin', 'ng2-admin', 'eladmin', 'adminpanel', 'admin-ui', 'admin-template', 'admin-panel',
                'admin-dashboard'],
               ['activerecord', 'activemq', 'activejob', 'active-record']

               ]

def same_things(string):
    for idx,i in enumerate(string_list):
        if string in i:
            string = target[idx]
    return string

# Create your views here.
def index(request):
    if request.method == 'POST':
        form = MyForm(request.POST)
        if form.is_valid():
            num = form.cleaned_data['clusterNum']
            org = form.cleaned_data['keyword']
        print(num)
        print(org)

        # org = ["aws","facebook","google","naver","kakao","apple","alibaba","tencent","baidu","microsoft","samsung"]
        org=org.split(" ")

        org_dic = {}
        for o in org:
            url = "https://github.com/orgs/{}/repositories".format(o)
            print("{} 에 대한 정보 수집 시작.".format(url))
            res = requests.get(url)
            try:
                res.raise_for_status()
            except:
                print("입력하신 기업\"{}\" 에 대한 정보가 존재하지 않습니다.\n".format(o))
                continue
            html = res.text
            soup = BeautifulSoup(html, "html.parser")
            try:
                max_page = int(soup.find("div", attrs={"role": "navigation"}).find_all("a")[-2].get_text())
            except:
                max_page = 1
            #     print(o,max_page)
            item_temp = []
            print("{}{}수집시작{}".format("*" * 10, o, "*" * 10))
            for p in range(1, max_page + 1):
                print("{}% 완료".format(round((p * 100 / max_page + 1))))
                time.sleep(1)
                url = "https://github.com/orgs/{}/repositories?page={}".format(o, p)
                res = requests.get(url)
                res.raise_for_status()
                html = res.text
                soup = BeautifulSoup(html, "html.parser")
                for item in soup.find("div", attrs={"class": "org-repos repo-list"}).find_all("li", attrs={"class": "Box-row"}):
                    #             print(item.a.get_text().strip())
                    item_temp.append(item.a.get_text().strip())
            org_dic[o] = item_temp
            print()

        print("수집완료.")

        excel_buffer1 = io.BytesIO() # 기업명.xlsx
        excel_buffer2 = io.BytesIO() # 기업명_vector.xlsx
        excel_buffer3 = io.BytesIO() # 기업명_kmeans_10clusters.xlsx
        vector_image_buffer = io.BytesIO() # 벡터 이미지 시각화
        excel_buffer4 = io.BytesIO() # DBSCAN_CLUSTER.XLSX

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            for i in org_dic:
                print("{}{}수집시작{}".format("*" * 10, i, "*" * 10))
                temp = []
                for idx, a in enumerate(org_dic[i]):
                    print("{}% Crawling 완료".format(round(idx * 100 / len(org_dic[i]))))
                    url = "https://github.com/{}/{}".format(i, a)
                    #html = res.text
                    #soup = BeautifulSoup(html, "html.parser")

                    try:
                        res = requests.get(url)
                        res.raise_for_status()
                        html = res.text
                        soup = BeautifulSoup(html, "html.parser")
                    except:
                        time.sleep(10)
                        res = requests.get(url)
                        #             res.raise_for_status()
                        html = res.text
                        soup = BeautifulSoup(html, "html.parser")

                        # 브랜치 수(파생 프로젝트 수)
                    try:
                        branch_num = soup.find("div", attrs={
                            "class": "flex-self-center ml-3 flex-self-stretch d-none d-lg-flex flex-items-center lh-condensed-ultra"}).strong.get_text()
                    except:
                        branch_num = 0

                        # 풀 리퀘스트 수(코드 병합 요구 수)
                    try:
                        pull_requests_num = soup.find("a", attrs={"id": "pull-requests-tab"}).find("span", attrs={
                            "class": "Counter"}).get_text()
                    except:
                        pull_requests_num = 0

                        # 커밋 수(수정 요구수)
                    try:
                        commit_num = soup.find("li", attrs={"class": "ml-0 ml-md-3"}).strong.get_text()
                    except:
                        commit_num = 0

                        # 스타 수(북마크 수, 인기도)
                    try:
                        star_num = soup.find("ul",
                                             attrs={"class": "pagehead-actions flex-shrink-0 d-none d-md-inline"}).find(
                            "a", attrs={"class": "social-count js-social-count"}).get_text()
                        star_num = star_num.replace('\t', '').replace('\n', '').strip()
                    except:
                        star_num = 0

                        # 토픽(개발언어의 주제)
                    try:
                        topic = soup.find("div", attrs={"class": "BorderGrid-cell"}).find_all("a", attrs={
                            "class": "topic-tag topic-tag-link"})
                        topics = [i.get_text().replace("\n", "").replace("\t", "").strip() for i in topic]
                    except:
                        topics = []

                        # 릴리즈 정보(버전 정보)
                    try:
                        release_info = soup.find("div", attrs={"class": "BorderGrid BorderGrid--spacious"}).find("span",
                                                                                                                 attrs={
                                                                                                                     "class": "css-truncate css-truncate-target text-bold mr-2"}).get_text().strip()
                    except:
                        release_info = ""

                    temp.append([a, branch_num, pull_requests_num, commit_num, star_num, topics, release_info])
                df = pd.DataFrame(temp,
                                      columns=["ProjectName", "BranchNum", "PullRequestNum", "CommitNum", "StarNum",
                                               "Topics", "ReleaseInfo"])

                df.to_excel(excel_buffer1, index=False)
                print("{}.xlsx".format(i), "저장 완료")
                # zip에 엑셀파일 추가
                zipf.writestr("{}.xlsx".format(i), excel_buffer1.getvalue())
                #여기까지는 ok


            # 기업 정보 벡터화
            for o in org:
                df = pd.read_excel(excel_buffer1.getvalue())
                for idx, i in enumerate(df["Topics"]):
                    if len(i) == 2:
                        df.drop(index=idx, inplace=True)
                df.reset_index()
                df.to_excel(excel_buffer1)
                result = []
                topic_dic = {}
                for idx, j in enumerate(df['Topics']):
                    i_list = j.replace("[", "").replace("]", "").replace("'", "").split(",")
                    i_list = [i.strip() for i in i_list]
                    df['Topics'][idx] = i_list
                    for i in i_list:
                        i = i.strip()
                        i = same_things(i)  # 위에서 만든 여러 토픽을 하나의 토픽으로 묶는 함수
                        if len(i) == 0:
                            continue
                        if i not in topic_dic:
                            topic_dic[i] = 1
                        else:
                            topic_dic[i] += 1

                topic_dic = sorted(topic_dic.items(), key=lambda x: x[1], reverse=True)

                topic_list = [i[0] for i in topic_dic]
                # 위 과정들을 통해 얻은 토픽들을 통해 칼럼을 만듬
                df_vector = pd.DataFrame(topic_list).transpose()

                topic_dic_df = {}
                for i in topic_list:
                    topic_dic_df[i] = 0
                # 토픽들을 칼럼으로 가지는 각각의 패키지 별로 벡터를 만들고 모든 성분을 0으로 채움
                df_vector = pd.DataFrame(result, index=df['ProjectName'], columns=topic_list).fillna(0)

                # 각 패키지 별로 가지고있는 토픽을 +1 씩 하여 각 패키지의 벡터를 생성함
                for idx, topics in enumerate(df['Topics']):
                    for topic in topics:
                        if topic in topic_list:
                            topic = same_things(topic)
                            df_vector.loc[df['ProjectName'].iloc[idx], topic] += 1

                df_vector.to_excel(excel_buffer2, index=False)
                print("{}_vectors.xlsx".format(o), "저장완료.")
                # zip에 엑셀파일 추가
                zipf.writestr("{}_vectors.xlsx".format(o), excel_buffer2.getvalue())
                #여기까지 ok

        # with open("{}.zip".format(org), "wb") as zip_file:
        #     zip_file.write(zip_buffer.getvalue())
        # return response

# 자 perplexity must be less than n_samples 발생 잠깐 보류

            # K-Means 클러스터링
            for o in org:
                print("{} {} K-Means 클러스터링 시작 {}".format("*" * 10, o, "*" * 10))
                # excel_name = "{}_vectors.xlsx".format(o)
                df_org = pd.read_excel(excel_buffer1.getvalue())
                df_vector = pd.read_excel(excel_buffer2.getvalue())
                '''
                데이터 개수가 적으면 perplexity Error가 발생할 수 있습니다 !
                이럴 경우에는 데이터의 개수(= 엑셀 파일의 행 개수)보다 1작은 값으로 perplexity 파라미터를 조정해주면 됩니다.
                '''
                # 원래 코드
                m = TSNE(learning_rate=50, random_state=0)  # perplexity Error가 발생할 경우 perplexity = 데이터의 개수 - 1로 파라미터 설정
                tsne_features = m.fit_transform(pd.DataFrame(df_vector))

                df = pd.DataFrame()
                df['x'] = tsne_features[:, 0]
                df['y'] = tsne_features[:, 1]
                n_clusters=num
                if len(df) <= n_clusters:
                    print("Repository 가 {}개 미만입니다.".format(n_clusters))
                    n_clusters = len(df) // 3
                    print("{}개의 Cluster로 군집화 시작합니다.".format(n_clusters))
                else:
                    # 10개의 Cluster 지정
                    n_clusters = n_clusters
                kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                # kmeans = KMeans(n_clusters=4,verbose = 1)

                # 모델에 fitting 하기
                kmeans.fit(tsne_features)

                y_kmeans = kmeans.predict(tsne_features)
                y_kmeans[0:10]  # 군집화 된 결과

                plt.scatter(np.array(tsne_features)[:, 0], np.array(tsne_features)[:, 1], c=y_kmeans, s=20,
                            cmap='viridis')

                centers = kmeans.cluster_centers_
                plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
                plt.grid()
                plt.title(excel_buffer2)

                plt.savefig(vector_image_buffer, format='png')
                plt.close()

                #zip에 이미지 추가
                zipf.writestr("{}_vector.png".format(o), vector_image_buffer.getvalue())

                # plt.show()
#

                clustered_dic = {}
                clustered_list = []
                cluster_num = len(set(y_kmeans))
                for idx, i in enumerate(y_kmeans):
                    if i not in clustered_dic:
                        clustered_dic[i] = [df_org['ProjectName'][idx]]
                    else:
                        clustered_dic[i].append(df_org['ProjectName'][idx])
                clustered_dic = sorted(clustered_dic.items(), key=lambda x: x[0])
                df_cluster = pd.DataFrame(clustered_dic, columns=['num', 'clusters'])
                cluster_num = [len(i) for i in df_cluster['clusters']]
                df_cluster['cluster_num'] = cluster_num

                # 군집화된 패키지들은 어떤 토픽들을 가지고 있을까?
                topic_clustered_list = []
                for i in df_cluster['clusters']:
                    temp_dic = {}
                    for j in i:
                        topics = list(
                            df_org[df_org['ProjectName'] == j]['Topics'].values[0].replace("[", "").replace("]", "").replace("'", "").split(","))
                        for i in topics:
                            if len(i) == 0:
                                continue
                            i = same_things(i.strip())
                            if i not in temp_dic:
                                temp_dic[i] = 1
                            else:
                                temp_dic[i] += 1
                    temp_dic = sorted(temp_dic.items(), key=lambda x: x[1], reverse=True)
                    topic_clustered_list.append(temp_dic[:15])
                df_cluster['top_15_topics'] = topic_clustered_list

                df_cluster.to_excel(excel_buffer3, index=False)
                print("{}_vectors.xlsx".format(o), "저장완료.")
                #zip에 엑셀파일 추가
                zipf.writestr("{}_kmeans_10clusters.xlsx".format(o), excel_buffer3.getvalue())

                # df_cluster.to_excel("{}_kmeans_10clusters.xlsx".format(o), index=False)
                # print(df_cluster)
                # print("{}_kmeans_10clusters.xlsx".format(o), "저장완료")
                # print("*" * 50)

            for o in org:
                print("{} {} DBSCAN 클러스터링 시작 {}".format("*" * 10, o, "*" * 10))
                df_org = pd.read_excel(excel_buffer1.getvalue())
                df_vector = pd.read_excel(excel_buffer2.getvalue())
                # eps 값을 조정해나가면서 클러스터링을 해야 더 정확한 결과가 나온다
                dbscan = DBSCAN(eps=0.3)
                dbscan_cluster = dbscan.fit_predict(df_vector)
                dbscan_cluster
                dbscan_clustered_dic = {}
                dbscan_clustered_list = []
                dbscan_cluster_num = len(set(dbscan_cluster))

                for idx, i in enumerate(dbscan_cluster):
                    if i not in dbscan_clustered_dic:
                        dbscan_clustered_dic[i] = [df_org['ProjectName'][idx]]
                    else:
                        dbscan_clustered_dic[i].append(df_org['ProjectName'][idx])

                # 클러스터링이 된 패키지들
                # 20개의 군집으로 생성

                dbscan_clustered_dic = sorted(dbscan_clustered_dic.items(), key=lambda x: x[0])

                df_dbscan_cluster = pd.DataFrame(dbscan_clustered_dic, columns=['num', 'clusters'])
                dbscan_cluster_num = [len(i) for i in df_dbscan_cluster['clusters']]
                df_dbscan_cluster['cluster_num'] = dbscan_cluster_num
                topic_dbscan_clustered_list = []
                for i in df_dbscan_cluster['clusters']:
                    temp_dic = {}
                    for j in i:
                        topics = df_org[df_org['ProjectName'] == j]['Topics'].values[0].replace("[", "").replace("]",
                                                                                                                 "").replace(
                            "'", "").strip().split(",")
                        for i in topics:
                            if len(i) == 0:
                                continue
                            i = same_things(i)
                            if i not in temp_dic:
                                temp_dic[i] = 1
                            else:
                                temp_dic[i] += 1
                    temp_dic = sorted(temp_dic.items(), key=lambda x: x[1], reverse=True)
                    #     print(temp_dic[:15]) # 상위 15개만 보여줌
                    #     print()
                    topic_dbscan_clustered_list.append(temp_dic[:15])
                df_dbscan_cluster['top_15_topics'] = topic_dbscan_clustered_list

                df_dbscan_cluster.to_excel(excel_buffer4, index=False)
                print("{}_DBSCAN_clusters.xlsx".format(o), "저장완료.")
                # zip에 엑셀파일 추가
                zipf.writestr("{}_DBSCAN_clusters.xlsx".format(o), excel_buffer4.getvalue())

        zip_buffer.seek(0)
        response = HttpResponse(zip_buffer, content_type='application/zip')
        # response = StreamingHttpResponse(
        #     FileWrapper(zip_buffer),
        #     content_type='application/zip'
        # )

        filename = form.cleaned_data['keyword'] + '.zip'  # 폼 데이터에서 파일 이름 가져오기
        response['Content-Disposition'] = f'attachment; filename="{filename}"'.format(org)

        return response

    else:
        form = MyForm()
        return render(request, 'ai_bigtech_company/index.html', {'form': form})