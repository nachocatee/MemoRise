package com.tjjhtjh.memorise.domain.memo.repository;

import com.tjjhtjh.memorise.domain.memo.repository.entity.Bookmark;

import java.awt.print.Book;
import java.util.List;
import java.util.Optional;

public interface BookmarkRepositoryCustom {
   Optional<Bookmark> findByMemoAndUser(Long memoId, String email);
   List<Bookmark> bookmarkExistCheck(Long memoId, Long userSeq);

}
