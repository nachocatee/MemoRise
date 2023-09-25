package com.tjjhtjh.memorise.domain.memo.repository;

import com.tjjhtjh.memorise.domain.memo.repository.entity.Bookmark;
import com.tjjhtjh.memorise.domain.memo.service.dto.response.MyMemoResponse;

import java.util.List;
import java.util.Optional;

public interface BookmarkRepositoryCustom {
   Optional<Bookmark> findByMemoAndUser(Long memoId, String email);
   List<Bookmark> bookmarkExistCheck(Long memoId, Long userSeq);
   List<MyMemoResponse> isBookmarkTrueList(Long userSeq);

}
