package com.tjjhtjh.memorise.domain.memo.repository;

import com.querydsl.jpa.impl.JPAQueryFactory;
import com.tjjhtjh.memorise.domain.memo.repository.entity.Bookmark;
import org.springframework.data.jpa.repository.support.QuerydslRepositorySupport;

import java.util.List;
import java.util.Optional;

import static com.tjjhtjh.memorise.domain.memo.repository.entity.QBookmark.bookmark;


public class BookmarkRepositoryImpl extends QuerydslRepositorySupport implements BookmarkRepositoryCustom {

    private final JPAQueryFactory queryFactory;

    public BookmarkRepositoryImpl(JPAQueryFactory jpaQueryFactory){
        super(Bookmark.class);
        this.queryFactory = jpaQueryFactory;
    }

    @Override
    public Optional<Bookmark> findByMemoAndUser(Long memoId, String email) {
        return queryFactory.selectFrom(bookmark)
                .where(bookmark.user.email.eq(email).and(bookmark.memo.memoSeq.eq(memoId)))
                .stream().findAny();
    }

    @Override
    public List<Bookmark> bookmarkExistCheck(Long memoId, Long userSeq) {
        return queryFactory.selectFrom(bookmark).where(bookmark.user.userSeq.eq(userSeq).and(bookmark.memo.memoSeq.eq(memoId)))
                .fetch();
    }
}
