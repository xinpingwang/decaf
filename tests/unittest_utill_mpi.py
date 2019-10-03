from decaf.util import mpi
import os
import unittest

_MPI_TEST_DIR = '/tmp/mpi_test_dir'
_MPI__DUMP_TEST_FILE = '/tmp/iceberk.test.unittest_mpi.dump.npy'


class TestMPI(unittest.TestCase):
    """
    Test the mpi module
    """

    def setUp(self) -> None:
        pass

    def testBasic(self):
        self.assertIsNotNone(mpi.COMM)
        self.assertLess(mpi.RANK, mpi.SIZE)
        self.assertIsInstance(mpi.HOST, str)

    def testMkdir(self):
        mpi.mkdir(_MPI_TEST_DIR)
        self.assertTrue(os.path.exists(_MPI_TEST_DIR))

    def testAnyAll(self):
        self.assertTrue(mpi.mpi_all(True))
        self.assertFalse(mpi.mpi_all(False))
        self.assertEqual(mpi.mpi_all(mpi.RANK == 0), mpi.SIZE == 1)
        self.assertTrue(mpi.mpi_any(True))
        self.assertFalse(mpi.mpi_any(False))
        self.assertTrue(mpi.mpi_any(mpi.RANK == 0))

    def testRootDecide(self):
        self.assertTrue(mpi.root_decide(True))
        self.assertFalse(mpi.root_decide(False))
        self.assertTrue(mpi.root_decide(mpi.RANK == 0))
        self.assertFalse(mpi.root_decide(mpi.RANK != 0))

    def testElect(self):
        result = mpi.elect()
        self.assertLess(result, mpi.SIZE)
        all_results = mpi.COMM.allgather(result)
        self.assertEqual(len(set(all_results)), 1)
        num_presidents = mpi.COMM.allreduce(mpi.is_president())
        self.assertEqual(num_presidents, 1)

    def testIsRoot(self):
        if mpi.RANK == 0:
            self.assertTrue(mpi.is_root())
        else:
            self.assertFalse(mpi.is_root())

    def testBarrier(self):
        import time
        # sleep for a while, and resume
        time.sleep(mpi.RANK)
        mpi.barrier()
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
