import absl
from absl import flags
from bleu_score import Bleu
from absl import app
from utils import input_from_file

FLAGS = flags.FLAGS

flags.DEFINE_integer('n', 4, "N gram")
flags.DEFINE_list('n_gram_weights', None, "weight on each n gram")
flags.DEFINE_string('cand_text', None, "candidate text")
flags.DEFINE_string("ref_text", None, 'reference text')
flags.DEFINE_string('cand_file', None, "candidate reference text file")
flags.DEFINE_string('ref_file', None, 'Reference filename')
flags.DEFINE_string('cand_file_dir', None, "candidate file directory")
flags.DEFINE_string('ref_file_dir', None, "reference file directory")

# flags.mark_flag_as_required("cand_text")
# flags.mark_flag_as_required("ref_text")


def main(argv):
  del argv

  if FLAGS.n_gram_weights is not None:
    assert len(FLAGS.n_gram_weights) == FLAGS.n

  assert FLAGS.ref_text or FLAGS.ref_file == True, "Either reference text or input file should be provided!" 
  assert FLAGS.cand_text or FLAGS.cand_file == True, "Either candidate text or input file should be provided!" 


  bleu_score = Bleu(FLAGS.n, FLAGS.n_gram_weights)
  bleu = bleu_score(FLAGS.cand_text, FLAGS.ref_text)

  print("bleu score: ", bleu)



if __name__ == '__main__':
  app.run(main)




