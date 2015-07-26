import json, os, binascii, jinja2
import scipy.spatial.distance as sp_dist
from sklearn.manifold import TSNE
import numpy as np

class MathUtil:
    @staticmethod
    def log_0(x):
        eps = 1e-37
        result = np.empty(x.shape)
        x_less_eps = x < eps
        x_more_esp = np.logical_not(x_less_eps)
        result[x_less_eps] = 0
        result[x_more_esp] = np.log(x[x_more_esp])
        return result

    @staticmethod
    def div_0(u, v):
        result = np.empty(v.shape)
        v_not_eq_0 = v != 0
        v_eq_0 = np.logical_not(v_not_eq_0)
        result[v_not_eq_0] = u[v_not_eq_0] / v[v_not_eq_0]
        result[v_eq_0] = 0
        return result

    @staticmethod
    def sym_kl_dist(u, v):
        s = (u + v) / 2.0
        p1 = MathUtil.div_0(u, s)
        log_p1 = MathUtil.log_0(p1)

        p2 = MathUtil.div_0(v, s)
        log_p2 = MathUtil.log_0(p2)

        return ((u * log_p1).sum() + (v * log_p2).sum()) / 2.0


class ArtmModelVisualizer(object):
    def __init__(self, model, num_top_tokens=30, dictionary_path=None, lambda_step=0.1):
        self.model = model
        self.num_top_tokens = num_top_tokens
        self.lambda_step = lambda_step
        self.dictionary_path = dictionary_path

        self.eps = 1e-37
        self.data = None

        if not os.path.exists('ldavis.js'):
            self._download_ldavis()

        if not self.model._initialized:
            print 'Model does not exist yet. Use ArtmModel.initialize()/ArtmModel.fit_*()'
            return

        if self.dictionary_path is not None:
            self.model_dictionary = library.Library().LoadDictionary(self.dictionary_path)
        else:
            print 'dictionary path is None, skip visualization.'
            return

        self.data = self._gen_data()

    def _gen_data(self):
        p_wt_model, phi = self.model.master.GetTopicModel(self.model._model)
        phi = phi.transpose()

        vocab = np.array(p_wt_model.token)

        dist_matrix = sp_dist.pdist(phi.transpose(), MathUtil.sym_kl_dist)

        tsne = TSNE(n_components=2)
        centers = tsne.fit_transform(sp_dist.squareform(dist_matrix)).transpose()

        topic_proportion = np.ones((self.model.num_topics,)) / self.model.num_topics

        term_frequency = np.array([entry.token_count * 1.0 for entry in self.model_dictionary.entry])[:, np.newaxis]
        term_proportion = term_frequency / np.sum(term_frequency)

        p_w = np.sum(phi, axis=1)[:, np.newaxis]
        term_topic_frequency = phi * (term_frequency / (p_w + self.eps))
        topic_given_term = phi / (p_w + self.eps)
        kernel = topic_given_term * MathUtil.log_0(topic_given_term)
        saliency = term_proportion * np.sum(kernel, axis=1)[:, np.newaxis]

        sorting_indices = saliency.ravel().argsort()
        default_terms = vocab[sorting_indices][:self.num_top_tokens]
        counts = term_frequency[:, np.newaxis][sorting_indices]

        rs = np.arange(self.num_top_tokens)[::-1]
        topic_str_list = ['Topic' + str(i + 1) for i in xrange(self.model.num_topics)]
        category = np.repeat(topic_str_list, self.num_top_tokens)
        topics = np.repeat(np.arange(self.model.num_topics), self.num_top_tokens)

        lift = phi / (term_proportion + self.eps)
        phi_column = phi.ravel()
        lift_column = lift.ravel()

        tinfo = dict()
        tinfo['Term'] = default_terms.tolist()
        tinfo['Category'] = ['Default'] * default_terms.size
        tinfo['logprob'] = rs.tolist()
        tinfo['loglift'] = rs.tolist()
        tinfo['Freq'] = counts.tolist()
        tinfo['Total'] = counts.tolist()

        term_indices = []
        topic_indices = []

        def find_relevance(ind):
            relevance = ind * MathUtil.log_0(phi) + (1 - ind) * MathUtil.log_0(lift)
            idx = np.apply_along_axis(lambda x: x.argsort()[:self.num_top_tokens], axis=0, arr=relevance)
            idx = idx.ravel()
            indices = np.concatenate((idx, topics))

            tinfo['Term'] += vocab[idx]
            tinfo['Category'] += category
            tinfo['logprob'] += np.round(MathUtil.log_0(phi_column[indices]), 4).tolist()
            tinfo['loglift'] += np.round(MathUtil.log_0(lift_column[indices]), 4).tolist()

            term_indices.extend(idx.tolist())
            topic_indices.extend(topics)

        for i in np.arange(0, 1, self.lambda_step):
            find_relevance(i)

        tinfo['Total'] += term_frequency[term_indices].tolist()
        for i in xrange(len(term_indices)):
            tinfo['Freq'].append(float(term_topic_frequency[term_indices[i], topic_indices[i]]))

        all_tokens = []
        all_topics = []
        all_values = []
        for token_index in xrange(len(p_wt_model.token)):
            for topic_index in xrange(self.model.num_topics):
                all_tokens.append(p_wt_model.token[token_index])
                all_topics.append(topic_index + 1)
                all_values.append(round(phi[token_index][topic_index], 8))

        data = {'mdsDat': {'x': centers[0].tolist(),
                           'y': centers[1].tolist(),
                           'topics': range(1, self.model.num_topics + 1),
                           'Freq': (topic_proportion * 100).tolist(),
                           'cluster': [1] * self.model.num_topics},
                'tinfo': {'Term': tinfo['Term'],
                          'logprob': tinfo['logprob'],
                          'loglift': tinfo['loglift'],
                          'Freq': tinfo['Freq'],
                          'Total': tinfo['Total'],
                          'Category': tinfo['Category']},
                'token.table': {'Term': all_tokens,
                                'Topic': all_topics,
                                'Freq': all_values},
                'R': self.num_top_tokens,
                'lambda.step': self.lambda_step,
                'plot.opts': {'xlab': 'PC-1',
                              'ylab': 'PC-2'},
                'topic_order': range(self.model.num_topics)}

        return data

    @staticmethod
    def _download_ldavis():
        import urllib2
        address = "https://raw.githubusercontent.com/romovpa/" + \
            "bigartm/notebook-ideas/notebooks/ldavis/ldavis.js"
        ldavis_js = urllib2.urlopen(address).read()
        with open('ldavis.js', 'w') as fout:
            fout.write(ldavis_js)

    def _generate_json(self):
        return json.dumps(self.data)
        
    def _repr_html_(self):
        random_figid = binascii.hexlify(os.urandom(16))
        html = TEMPLATE_NOTEBOOK.render(
            figid=random_figid,
            figure_json=self._generate_json(),
            d3_url=URL_D3,
            ldavis_url='ldavis.js',
            extra_css=LDAVIS_CSS,
        )
        return html
        
    def to_file(self, filename, title=None):
        if self.data is None:
            print "Data wasn't generated."
            return

        if title is None:
            title = 'LDAvis Topic Model Visualization'
            
        with open('ldavis.js') as f:
            js_code = f.read()
            
        html = TEMPLATE_PAGE.render(
            title=title,
            d3_url=URL_D3,
            ldavis_url='ldavis.js',
            data_json=self._generate_json(),
            extra_css=LDAVIS_CSS,
            js_code=js_code,
        )
        with open(filename, 'wt') as fout:
            fout.write(html)

URL_D3 = 'https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.js'

LDAVIS_CSS = """
path {
  fill: none;
  stroke: none;
}

.xaxis .tick.major {
    fill: black;
    stroke: black;
    stroke-width: 0.1;
    opacity: 0.7;
}

.slideraxis {
    fill: black;
    stroke: black;
    stroke-width: 0.4;
    opacity: 1;
}

text {
    font-family: sans-serif;
    font-size: 11px;
}
"""

TEMPLATE_NOTEBOOK = jinja2.Template("""
<div id="ldavis_{{figid}}"></div>

<style>{{ extra_css }}</style>

<script>
function ldavis_load_lib(url, callback){
  var s = document.createElement('script');
  s.src = url;
  s.async = true;
  s.onreadystatechange = s.onload = callback;
  s.onerror = function(){console.warn("failed to load library " + url);};
  document.getElementsByTagName("head")[0].appendChild(s);
}

figure_data = {{ figure_json }};

if(typeof(LDAvis) !== "undefined"){
   // already loaded: just create the figure
   var vis = new LDAvis("#ldavis_{{figid}}", figure_data);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/mpld3
   require.config({paths: {d3: "{{ d3_url[:-3] }}"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      ldavis_load_lib("{{ ldavis_url }}", function(){
         var vis = new LDAvis("#ldavis_{{figid}}", figure_data);
      });
    });
}else{
    // require.js not available: dynamically load d3 & mpld3
    ldavis_load_lib("{{ d3_url }}", function(){
         ldavis_load_lib("{{ ldavis_url }}", function(){
                 var vis = new LDAvis("#ldavis_{{figid}}", figure_data);
            })
         });

}
</script>
""")


TEMPLATE_PAGE = jinja2.Template("""
<!DOCTYPE html>
<html>
  <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <title>{{ title }}</title>
    <script src="{{ d3_url }}"></script>
    <script>{{ js_code }}</script>
    <style>
    {{ extra_css }}
    </style>
  </head>

  <body>
    <div id = "lda"></div>
    <script>
      data = {{ data_json }};
      var vis = new LDAvis("#lda", data);
    </script>
  </body>

</html>

""")


import sys
HOME = '/home/vovapolu/Projects/'
BIGARTM_PATH = HOME + 'bigartm/'
BIGARTM_BUILD_PATH = BIGARTM_PATH + 'build/'
sys.path.append(os.path.join(BIGARTM_PATH, 'src/python'))
os.environ['ARTM_SHARED_LIBRARY'] = os.path.join(BIGARTM_BUILD_PATH, 'src/artm/libartm.so')

import artm.artm_model
from artm.artm_model import *

def test():
    model = ArtmModel(num_topics=100)
    if len(glob.glob('kos' + "/*.batch")) < 1:
        parse(data_path='', data_format='bow_uci', collection_name='kos')
    model.load_dictionary(dictionary_name='dictionary', dictionary_path='kos/dictionary')
    model.initialize(dictionary_name='dictionary')
    model.regularizers.add(SmoothSparsePhiRegularizer(name='SparsePhi', tau=-1.5))
    model.regularizers.add(SmoothSparseThetaRegularizer(name='SparseTheta', tau=-5.0))
    model.regularizers.add(DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=100000.0))
    model.fit_offline(data_path='kos', num_collection_passes=15)

    vis = ArtmModelVisualizer(model, dictionary_path="kos/dictionary")
    vis.to_file("test.html")

test()