package com.tjjhtjh.memorise.domain.memo.repository;

import com.tjjhtjh.memorise.domain.memo.repository.entity.Bookmark;

import java.util.Optional;

public interface BookmarkRepositoryCustom {
   Optional<Bookmark> findByMemoAndUser(Long memoId, String email);

}
